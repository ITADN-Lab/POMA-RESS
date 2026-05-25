"""
2.3 Blind diagnostic test: Run 100 steps of AdamW, compute kappa and rho_g,
predict whether NS will help, then run both AdamW and NS for full training.
Tests on C-MAPSS FD002 and FD003 (diagnostic thresholds calibrated on FD001/CWRU/CIFAR-100).
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))
from train_cmapss_rul import (CMAPSSDataset, LSTM_RUL, evaluate, compute_kappa,
                              build_optimizer, parse_args as cmapss_parse_args)


def compute_step_rho_g(optimizer, model, train_loader, device, n_steps=100):
    """Run n_steps of AdamW, compute inter-step gradient correlation rho_g
    and final momentum condition number kappa."""
    criterion = nn.MSELoss()
    model.train()

    prev_grad_flat = None
    rho_values = []

    loader_iter = iter(train_loader)
    for step in range(n_steps):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            x, y = next(loader_iter)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()

        # Collect flattened gradient
        grads = []
        for p in model.parameters():
            if p.grad is not None and p.dim() >= 2:
                grads.append(p.grad.detach().flatten()[:2000])
        if grads:
            cur_grad = torch.cat(grads)

        # No grad clipping during diagnostic phase
        optimizer.step()

        if prev_grad_flat is not None and grads:
            min_len = min(len(cur_grad), len(prev_grad_flat))
            c = cur_grad[:min_len]
            p = prev_grad_flat[:min_len]
            rho = (c @ p) / (c.norm() * p.norm() + 1e-8)
            rho_values.append(rho.item())

        if grads:
            prev_grad_flat = cur_grad.clone()

    rho_g = np.mean(rho_values) if rho_values else 0.0
    kappa = compute_kappa(optimizer)

    return rho_g, kappa, rho_values


def diagnostic_predict(rho_g, kappa):
    """Apply dual diagnostic: kappa>50 OR rho_g>0.8 -> NS recommended."""
    trigger_kappa = kappa > 50
    trigger_rho = rho_g > 0.8
    recommend_ns = trigger_kappa or trigger_rho
    reasons = []
    if trigger_kappa:
        reasons.append(f"kappa={kappa:.1f}>50")
    if trigger_rho:
        reasons.append(f"rho_g={rho_g:.3f}>0.8")
    if not reasons:
        reasons.append(f"kappa={kappa:.1f}<=50, rho_g={rho_g:.3f}<=0.8")
    return recommend_ns, reasons


def run_full_training(subset, optimizer_name, seed, data_dir, save_dir, ns_interval=100):
    """Run full 100-epoch training and return best RMSE."""
    # Parse args similar to main script
    import subprocess
    tag = f"cmapss_{subset}_{optimizer_name}_seed{seed}_blind"
    out_path = os.path.join(save_dir, f"{tag}.json")

    # Build model and data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)

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

    if optimizer_name == 'LAKTJU_NS':
        from optimizer.LAKTJU_NS import LAKTJU_NS
        optimizer = LAKTJU_NS(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                              weight_decay=1e-4, ns_interval=ns_interval, ns_steps=1, ns_max_dim=256)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_test_rmse = float('inf')

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

    result = {
        'subset': subset, 'optimizer': optimizer_name, 'seed': seed,
        'best_test_rmse': best_test_rmse, 'best_test_mae': best_test_mae,
    }
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=float)

    return best_test_rmse, best_test_mae


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='../data/cmapss')
    p.add_argument('--save_dir', default='../results/blind_test_2_3')
    p.add_argument('--subsets', nargs='+', default=['FD002', 'FD003'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    report = {}

    for subset in args.subsets:
        print(f"\n{'='*60}")
        print(f"Blind test: {subset}")
        print(f"{'='*60}")

        # Step 1: Run 100 steps of AdamW to get diagnostics
        print(f"\n[Phase 1] Diagnostic phase: 100 steps of AdamW")
        torch.manual_seed(42)
        np.random.seed(42)

        train_ds = CMAPSSDataset(args.data_dir, subset=subset, window_size=30, mode='train')
        test_ds = CMAPSSDataset(args.data_dir, subset=subset, window_size=30, mode='test')
        n_val = int(0.15 * len(train_ds))
        train_ds_diag, _ = torch.utils.data.random_split(
            train_ds, [len(train_ds) - n_val, n_val],
            generator=torch.Generator().manual_seed(42))
        train_loader_diag = DataLoader(train_ds_diag, batch_size=256, shuffle=True, num_workers=2)

        input_dim = test_ds.sensor_cols.__len__() if hasattr(test_ds, 'sensor_cols') else train_ds_diag.dataset.sensor_cols.__len__()

        model_diag = LSTM_RUL(input_dim=input_dim).to(device)
        opt_diag = torch.optim.AdamW(model_diag.parameters(), lr=1e-3, weight_decay=1e-4)

        rho_g, kappa, rho_history = compute_step_rho_g(
            opt_diag, model_diag, train_loader_diag, device, n_steps=100)

        # Step 2: Apply diagnostic
        recommend_ns, reasons = diagnostic_predict(rho_g, kappa)
        print(f"  rho_g = {rho_g:.4f}")
        print(f"  kappa = {kappa:.1f}")
        print(f"  Diagnostic: {'NS RECOMMENDED' if recommend_ns else 'NS NOT RECOMMENDED'}")
        for r in reasons:
            print(f"    - {r}")

        # Step 3: Run full training for BOTH optimizers to check ground truth
        print(f"\n[Phase 2] Full training: AdamW vs LAKTJU_NS ({len(args.seeds)} seeds)")

        results = {'subset': subset, 'diagnostic': {
            'rho_g': float(rho_g), 'kappa': float(kappa),
            'recommend_ns': recommend_ns, 'reasons': reasons,
            'rho_history': [float(v) for v in rho_history],
        }, 'seeds': {}}

        for seed in args.seeds:
            print(f"  seed={seed}: ", end='', flush=True)
            aw_rmse, aw_mae = run_full_training(
                subset, 'AdamW', seed, args.data_dir, args.save_dir)
            print(f"AdamW={aw_rmse:.2f}  ", end='', flush=True)
            ns_rmse, ns_mae = run_full_training(
                subset, 'LAKTJU_NS', seed, args.data_dir, args.save_dir)
            print(f"NS={ns_rmse:.2f}  delta={ns_rmse-aw_rmse:+.2f}")

            results['seeds'][str(seed)] = {
                'AdamW_rmse': aw_rmse, 'NS_rmse': ns_rmse,
                'delta': ns_rmse - aw_rmse,
            }

        # Step 4: Compute ground truth and diagnostic accuracy
        ns_wins = sum(1 for s in results['seeds'].values() if s['delta'] < 0)
        ns_losses = sum(1 for s in results['seeds'].values() if s['delta'] >= 0)
        mean_delta = np.mean([s['delta'] for s in results['seeds'].values()])

        print(f"\n[Phase 3] Verdict:")
        print(f"  NS wins: {ns_wins}/{len(args.seeds)} seeds")
        print(f"  Mean delta: {mean_delta:+.2f}")
        print(f"  Ground truth: {'NS HELPS' if mean_delta < 0 else 'NS NEUTRAL/HURTS'}")
        print(f"  Diagnostic prediction: {'NS RECOMMENDED' if recommend_ns else 'NS NOT RECOMMENDED'}")
        print(f"  Diagnostic correct: {((mean_delta < 0) == recommend_ns)}")

        results['verdict'] = {
            'ns_wins': ns_wins, 'ns_losses': ns_losses,
            'mean_delta': float(mean_delta),
            'ns_helps': mean_delta < 0,
            'diagnostic_correct': (mean_delta < 0) == recommend_ns,
        }

        report[subset] = results

    # Save full report
    report_path = os.path.join(args.save_dir, 'blind_test_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=float)

    # Summary
    print(f"\n{'='*60}")
    print("BLIND TEST SUMMARY")
    print(f"{'='*60}")
    for subset, r in report.items():
        d = r['diagnostic']
        v = r['verdict']
        print(f"\n{subset}:")
        print(f"  rho_g={d['rho_g']:.4f} kappa={d['kappa']:.1f}")
        print(f"  Predicted: {'NS' if d['recommend_ns'] else 'No NS'}")
        print(f"  Actual:    NS delta={v['mean_delta']:+.2f} ({v['ns_wins']}/{len(args.seeds)} seeds)")
        print(f"  Correct:   {v['diagnostic_correct']}")
    print(f"\nReport saved to {report_path}")


if __name__ == '__main__':
    main()
