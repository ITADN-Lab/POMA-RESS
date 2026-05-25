"""
Edge Deployment Benchmark for LafTJU-TII Phase 3.
Measures speed, memory, and accuracy for AdamW, LAKTJU-NS, LAKTJU-Lite on CIFAR-100 ResNet-18.
Self-contained; uses torchvision.
"""
import os, sys, time, json, argparse, numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision, torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
N_MEASURE = 300  # measure steps for speed (after warmup)


def get_dataloaders(data_dir='./dataset'):
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_ds = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=tf_train)
    test_ds = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=tf_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


def get_model():
    model = torchvision.models.resnet18(num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def benchmark_speed_memory(name, train_loader):
    """Measure ms/step, sec/epoch, and peak GPU memory for one optimizer."""
    torch.manual_seed(42)
    model = get_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    if name == 'AdamW':
        opt = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=5e-4)
    elif name == 'LAKTJU_NS':
        from optimizer.LAKTJU_NS import LAKTJU_NS
        opt = LAKTJU_NS(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                        weight_decay=5e-4, ns_interval=500, ns_steps=1, ns_max_dim=256, min_ndim=2)
    elif name == 'LAKTJU_Lite':
        from optimizer.LAKTJU_Lite import LAKTJU_Lite
        opt = LAKTJU_Lite(model.parameters(), lr=1e-3, beta1=0.9, beta2=0.999,
                          weight_decay=5e-4, total_steps=N_MEASURE, kf_update_interval=20)
    else:
        raise ValueError(name)

    if DEVICE == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    times = []
    for i, (x, y) in enumerate(train_loader):
        if i >= N_MEASURE:
            break
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()

        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        if i >= 10:  # skip warmup
            times.append((t1 - t0) * 1000)

    ms_per_step = np.mean(times)
    steps_per_epoch = 50000 // BATCH_SIZE  # ~391
    sec_per_epoch = ms_per_step * steps_per_epoch / 1000

    if DEVICE == 'cuda':
        mem_mb = torch.cuda.max_memory_allocated() / 1024**2
    else:
        import resource
        mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    # Get parameter count and optimizer state size
    n_params = sum(p.numel() for p in model.parameters())
    opt_state_mb = 0
    for group in opt.param_groups:
        for p in group['params']:
            state = opt.state.get(p, {})
            for v in state.values():
                if isinstance(v, torch.Tensor):
                    opt_state_mb += v.numel() * v.element_size()

    return ms_per_step, sec_per_epoch, mem_mb, n_params, opt_state_mb / 1024**2


def quick_train(name, train_loader, test_loader, epochs=5):
    """Quick training to measure accuracy for the overhead table."""
    torch.manual_seed(42)
    model = get_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    if name == 'AdamW':
        opt = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=5e-4)
        sched = lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    elif name == 'LAKTJU_NS':
        from optimizer.LAKTJU_NS import LAKTJU_NS
        opt = LAKTJU_NS(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                        weight_decay=5e-4, ns_interval=500, ns_steps=1, ns_max_dim=256, min_ndim=2)
        sched = lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    elif name == 'LAKTJU_Lite':
        from optimizer.LAKTJU_Lite import LAKTJU_Lite
        total_steps = epochs * (50000 // BATCH_SIZE)
        opt = LAKTJU_Lite(model.parameters(), lr=1e-3, beta1=0.9, beta2=0.999,
                          weight_decay=5e-4, total_steps=total_steps)
        sched = lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            total += x.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./dataset')
    parser.add_argument('--mode', default='full',
                        choices=['speed', 'train', 'full'])
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Loading CIFAR-100...")
    train_loader, test_loader = get_dataloaders(args.data_dir)

    print(f"\n{'='*70}")
    print(f"Phase 3: Edge Deployment Benchmark")
    print(f"{'='*70}")

    optimizers = ['AdamW', 'LAKTJU_NS', 'LAKTJU_Lite']
    results = {}

    # 1. Speed + Memory benchmark
    print(f"\n--- Speed & Memory (first {N_MEASURE} steps) ---")
    for name in optimizers:
        print(f"\nBenchmarking {name}...", flush=True)
        ms, sec, mem, n_params, opt_mem = benchmark_speed_memory(name, train_loader)
        results[name] = {'ms': ms, 'sec': sec, 'mem_mb': mem,
                         'n_params': n_params, 'opt_state_mb': opt_mem}

        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    # 2. Quick training accuracy
    print(f"\n--- Quick Training Accuracy (5 epochs) ---")
    for name in optimizers:
        print(f"Training {name}...", flush=True)
        acc = quick_train(name, train_loader, test_loader, epochs=5)
        results[name]['acc_5ep'] = acc
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    # 3. Summary
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")
    adam_ms = results['AdamW']['ms']
    adam_mem = results['AdamW']['mem_mb']
    print(f"{'Method':<15} {'ms/step':>8} {'s/epoch':>8} {'Mem(MB)':>9} {'OptState':>9} {'Slowdown':>10} {'MemΔ':>8} {'Acc@5':>8}")
    print(f"{'-'*80}")
    for name in optimizers:
        r = results[name]
        slowdown = f"+{(r['ms']/adam_ms - 1)*100:.1f}%"
        mem_delta = f"+{r['mem_mb'] - adam_mem:.0f}MB"
        print(f"{name:<15} {r['ms']:>8.2f} {r['sec']:>8.1f} {r['mem_mb']:>9.0f} "
              f"{r['opt_state_mb']:>9.1f} {slowdown:>10} {mem_delta:>8} {r['acc_5ep']*100:>7.2f}%")

    # Save results
    out = {
        'device': str(DEVICE),
        'batch_size': BATCH_SIZE,
        'results': {k: {kk: float(vv) if isinstance(vv, (np.floating, float, int)) else vv
                        for kk, vv in v.items()}
                     for k, v in results.items()}
    }
    os.makedirs('../results', exist_ok=True)
    with open('../results/edge_benchmark.json', 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nResults saved to ../results/edge_benchmark.json")


if __name__ == '__main__':
    main()
