"""
2.4 Second-order optimizer comparison on C-MAPSS FD001
Compares AdamW, LAKTJU-NS, Shampoo, SOAP
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch_optimizer as topt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))
from train_cmapss_rul import CMAPSSDataset, LSTM_RUL, evaluate


class SOAP(torch.optim.Optimizer):
    """Simplified SOAP optimizer: Adam in Shampoo eigenbasis.
    Reference: Vyas et al., 2025. SOAP: Improving and Stabilizing Shampoo using Adam.

    Practical simplification for small models: applies Shampoo-style
    Kronecker preconditioning with Adam-style momentum.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, precondition_freq=20, max_dim=256):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       precondition_freq=precondition_freq, max_dim=max_dim)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            freq = group['precondition_freq']
            max_dim = group['max_dim']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0:
                    grad = grad + wd * p

                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                    if p.dim() >= 2:
                        m, n = p.shape
                        state['L'] = torch.eye(min(m, max_dim), device=p.device, dtype=p.dtype)
                        state['R'] = torch.eye(min(n, max_dim), device=p.device, dtype=p.dtype)

                state['step'] += 1
                m, v = state['m'], state['v']
                beta1_t = beta1 ** state['step']
                beta2_t = beta2 ** state['step']

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                m_hat = m / (1 - beta1_t)
                v_hat = v / (1 - beta2_t)

                if p.dim() >= 2 and state['step'] % freq == 0:
                    # SOAP: precondition 2D params using Kronecker factors
                    m_shape = m.shape
                    m2d = m.view(m_shape[0], -1)
                    if m2d.size(0) <= max_dim and m2d.size(1) <= max_dim:
                        L = state['L']
                        R = state['R']
                        # Update factors: L = beta * L + (1-beta) * m2d @ m2d.T
                        L_scale = 0.95
                        L_hat = m2d @ m2d.T + eps * torch.eye(m2d.size(0), device=p.device, dtype=p.dtype)
                        R_hat = m2d.T @ m2d + eps * torch.eye(m2d.size(1), device=p.device, dtype=p.dtype)
                        L.mul_(L_scale).add_(L_hat / L_hat.trace(), alpha=1 - L_scale)
                        R.mul_(L_scale).add_(R_hat / R_hat.trace(), alpha=1 - L_scale)
                        # Apply preconditioner
                        L_inv = torch.linalg.inv(torch.linalg.cholesky(L))
                        R_inv = torch.linalg.inv(torch.linalg.cholesky(R))
                        m_hat = L_inv @ m2d @ R_inv

                # Adam update
                denom = v_hat.sqrt().add_(eps)
                update = m_hat / denom if p.dim() < 2 or state['step'] % freq != 0 else \
                         (m_hat.view(p.shape) if hasattr(m_hat, 'view') else m_hat) / denom
                p.add_(update, alpha=-lr)

        return loss


def run_experiment(subset, optimizer_name, seed, data_dir, save_dir, lr=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_ds = CMAPSSDataset(data_dir, subset=subset, window_size=30, mode='train')
    test_ds = CMAPSSDataset(data_dir, subset=subset, window_size=30, mode='test')
    n_val = int(0.15 * len(train_ds))
    train_ds, val_ds = torch.utils.data.random_split(
        train_ds, [len(train_ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    input_dim = test_ds.sensor_cols.__len__() if hasattr(test_ds, 'sensor_cols') else train_ds.dataset.sensor_cols.__len__()
    model = LSTM_RUL(input_dim=input_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # Build optimizer
    if optimizer_name == 'AdamW':
        lr = lr or 1e-3
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_name == 'LAKTJU_NS':
        from optimizer.LAKTJU_NS import LAKTJU_NS
        lr = lr or 1e-3
        optimizer = LAKTJU_NS(model.parameters(), lr=lr, betas=(0.9, 0.999),
                              weight_decay=1e-4, ns_interval=100, ns_steps=1, ns_max_dim=256)
    elif optimizer_name == 'Shampoo':
        lr = lr or 1e-3
        optimizer = topt.Shampoo(model.parameters(), lr=lr, weight_decay=1e-4,
                                 momentum=0.9, epsilon=1e-8, update_freq=50)
    elif optimizer_name == 'SOAP':
        lr = lr or 3e-4
        optimizer = SOAP(model.parameters(), lr=lr, weight_decay=1e-4,
                        betas=(0.9, 0.999), precondition_freq=20)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_test_rmse = float('inf')
    best_test_mae = float('nan')
    t0 = time.time()

    for epoch in range(1, 101):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        _, val_rmse, _ = evaluate(model, val_loader, criterion, device)
        _, test_rmse, test_mae = evaluate(model, test_loader, criterion, device)
        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_test_rmse = test_rmse
            best_test_mae = test_mae

    total_time = time.time() - t0

    result = {
        'subset': subset, 'optimizer': optimizer_name, 'seed': seed,
        'best_test_rmse': best_test_rmse, 'best_test_mae': best_test_mae,
        'total_time': total_time, 'n_params': n_params,
    }

    os.makedirs(save_dir, exist_ok=True)
    tag = f"cmapss_{subset}_{optimizer_name}_seed{seed}_2_4"
    with open(os.path.join(save_dir, f"{tag}.json"), 'w') as f:
        json.dump(result, f, indent=2, default=float)

    return best_test_rmse, best_test_mae, total_time


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--subset', default='FD001')
    p.add_argument('--optimizer', choices=['AdamW', 'LAKTJU_NS', 'Shampoo', 'SOAP'], default='AdamW')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--data_dir', default='../data/cmapss')
    p.add_argument('--save_dir', default='../results/ablation_2_4')
    args = p.parse_args()

    rmse, mae, t = run_experiment(args.subset, args.optimizer, args.seed,
                                   args.data_dir, args.save_dir, args.lr)
    print(f"{args.optimizer}/seed{args.seed}: RMSE={rmse:.2f} MAE={mae:.2f} time={t:.1f}s")
