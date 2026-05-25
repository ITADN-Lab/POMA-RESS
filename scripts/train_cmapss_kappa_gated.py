"""C-MAPSS RUL --- per-kernel kappa-gated PMO validation.

Validates that per-kernel kappa-gated PMO preserves the C-MAPSS win. On
C-MAPSS every LSTM kernel fails the kappa screen (kappa >> 1e4), so the gate's
skip-list is empty and kappa-gated PMO is bit-identical to plain PMO-all ---
the gate measurement reads optimizer state without consuming RNG or modifying
parameters. The TEP-forecasting case (where the recurrent kernels stay
kappa ~ 1e3) is the contrasting regime where the gate would skip them.

Modes:
  --optimizer AdamW                    -> baseline
  --optimizer LAKTJU_NS                -> PMO-all (NS on every kernel)
  --optimizer LAKTJU_NS --kappa_gate   -> PMO-kappa-gated: after the 100-step
        warm-up (just before the optimizer's first NS event) measure per-kernel
        kappa(M_t); kernels with kappa < threshold are excluded from NS.

Protocol mirrors train_cmapss_rul.py main(): 85/15 val split, validation-
selected best_test, ReduceLROnPlateau, MSE, window 30, 2-layer LSTM-128.
"""
import os, sys, json, time, argparse, datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS, '..', '..'))   # repo root for optimizer.*
sys.path.insert(0, os.path.join(_THIS, '..'))
sys.path.insert(0, _THIS)
from train_cmapss_rul import CMAPSSDataset, LSTM_RUL, evaluate


def kappa_of(M):
    """True condition number sigma_max/sigma_min of a (reshaped-2D) buffer."""
    m2d = M.detach().reshape(M.shape[0], -1).double()
    s = torch.linalg.svdvals(m2d)
    s = s[s > 1e-10]
    if s.numel() < 2:
        return None
    return (s[0] / s[-1]).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default=os.path.join(_THIS, '..', 'data', 'cmapss'))
    p.add_argument('--subset', required=True, choices=['FD001', 'FD002', 'FD003', 'FD004'])
    p.add_argument('--optimizer', required=True, choices=['AdamW', 'LAKTJU_NS'])
    p.add_argument('--kappa_gate', action='store_true')
    p.add_argument('--kappa_threshold', type=float, default=1e4)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--window_size', type=int, default=30)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--num_layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--grad_clip', type=float, default=0.0)
    p.add_argument('--ns_interval', type=int, default=100)
    p.add_argument('--ns_steps', type=int, default=1)
    p.add_argument('--ns_max_dim', type=int, default=256)
    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--save_dir', default=os.path.join(_THIS, '..', 'results_cmapss_kgate'))
    p.add_argument('--tag_suffix', default='')
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = CMAPSSDataset(args.data_dir, subset=args.subset,
                             window_size=args.window_size, mode='train')
    test_ds = CMAPSSDataset(args.data_dir, subset=args.subset,
                            window_size=args.window_size, mode='test')
    n_train = len(train_ds); n_val = int(0.15 * n_train); n_tr = n_train - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        train_ds, [n_tr, n_val],
        generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    input_dim = len(train_ds.dataset.sensor_cols)
    model = LSTM_RUL(input_dim=input_dim, hidden_dim=args.hidden_dim,
                     num_layers=args.num_layers, dropout=args.dropout).to(device)

    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      betas=(args.beta1, 0.999),
                                      weight_decay=args.weight_decay)
    else:
        from optimizer.LAKTJU_NS import LAKTJU_NS
        optimizer = LAKTJU_NS(model.parameters(), lr=args.lr,
                              betas=(args.beta1, 0.999),
                              weight_decay=args.weight_decay,
                              ns_interval=args.ns_interval, ns_steps=args.ns_steps,
                              ns_max_dim=args.ns_max_dim)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                               patience=10, factor=0.5)
    criterion = nn.MSELoss()

    named2d = [(n, pp) for n, pp in model.named_parameters() if pp.ndim >= 2]
    use_gate = args.kappa_gate and args.optimizer == 'LAKTJU_NS'
    gate_done = not use_gate
    kappa_gate_log = None
    global_step = 0
    best_val = float('inf'); best_test = float('inf'); best_mae = float('nan')
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # kappa-gate screen: fire just before the optimizer's first NS event
            if (not gate_done) and global_step == args.ns_interval - 1:
                klog, skip = {}, []
                for n, pp in named2d:
                    st = optimizer.state.get(pp, {})
                    k = kappa_of(st['exp_avg']) if 'exp_avg' in st else None
                    klog[n] = k
                    if k is not None and k < args.kappa_threshold:
                        skip.append(pp)
                optimizer._ns_skip_ids = set(id(pp) for pp in skip)
                kappa_gate_log = {
                    'measured_at_step': global_step + 1,
                    'kappa_threshold': args.kappa_threshold,
                    'per_kernel_kappa': klog,
                    'skip_list': [n for n, pp in named2d
                                  if id(pp) in optimizer._ns_skip_ids],
                    'n_kernels_2d': len(named2d), 'n_skipped': len(skip),
                }
                gate_done = True
                print(f"  [kappa-gate @ step {global_step+1}] "
                      f"skipped {len(skip)}/{len(named2d)} kernels: "
                      f"{kappa_gate_log['skip_list']}")
            optimizer.step()
            global_step += 1

        _, val_rmse, _ = evaluate(model, val_loader, criterion, device)
        _, test_rmse, test_mae = evaluate(model, test_loader, criterion, device)
        scheduler.step(val_rmse)
        if val_rmse < best_val:
            best_val = val_rmse; best_test = test_rmse; best_mae = test_mae
        history.append({'epoch': epoch, 'val_rmse': float(val_rmse),
                        'test_rmse': float(test_rmse)})
        if epoch % 20 == 0 or epoch == args.epochs:
            print(f"[{epoch:3d}/{args.epochs}] val={val_rmse:.3f} "
                  f"test={test_rmse:.3f} best_test={best_test:.3f} "
                  f"({time.time()-t0:.0f}s)")

    mode = args.optimizer
    if args.optimizer == 'LAKTJU_NS':
        mode = 'PMO_kgated' if use_gate else 'PMO_all'
    os.makedirs(args.save_dir, exist_ok=True)
    tag = f"cmapss_{args.subset}_{mode}_seed{args.seed}"
    if args.tag_suffix:
        tag += f"_{args.tag_suffix}"
    res = {
        'config': vars(args), 'mode': mode,
        'best_test_rmse': float(best_test), 'best_test_mae': float(best_mae),
        'best_val_rmse': float(best_val),
        'final_test_rmse': float(history[-1]['test_rmse']),
        'kappa_gate_log': kappa_gate_log,
        'history': history, 'total_time': time.time() - t0,
        'timestamp': datetime.datetime.now().isoformat(),
    }
    with open(os.path.join(args.save_dir, tag + '.json'), 'w') as f:
        json.dump(res, f, indent=2, default=float)
    print(f"saved {tag}.json | mode={mode} best_test_rmse={best_test:.4f}")


if __name__ == '__main__':
    main()
