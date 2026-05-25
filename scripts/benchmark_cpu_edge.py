"""
W3.2: CPU Edge Deployment Benchmark.
Simulates edge inference on CPU with batch_size=1.
Measures: inference latency, memory (RSS), model+optimizer state for training.
"""
import os, sys, time, json, argparse, resource, numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

DEVICE = torch.device('cpu')
BATCH_SIZE = 1  # edge inference simulation
N_WARMUP = 20
N_MEASURE = 100


class SyntheticDataset(torch.utils.data.Dataset):
    """Synthetic CIFAR-100-like data for CPU edge benchmarking (no torchvision needed)."""
    def __init__(self, n=1000, classes=100):
        self.data = torch.randn(n, 3, 32, 32)
        self.labels = torch.randint(0, classes, (n,))

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i], self.labels[i]


class ResNet18Proxy(nn.Module):
    """Lightweight CNN matching ResNet18 parameter count (~11.2M) for edge benchmarking."""
    def __init__(self, num_classes=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            self._block(64, 64, 2), self._block(64, 128, 2), self._block(128, 256, 2), self._block(256, 512, 2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, num_classes),
        )
    def _block(self, cin, cout, stride):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1), nn.BatchNorm2d(cout), nn.ReLU(),
            nn.Conv2d(cout, cout, 3, 1, 1), nn.BatchNorm2d(cout), nn.ReLU(),
            nn.Conv2d(cout, cout, 1, 1, 0) if cin != cout or stride != 1 else nn.Identity(),
        )
    def forward(self, x): return self.net(x)


def get_dataloader(data_dir='./dataset'):
    return torch.utils.data.DataLoader(SyntheticDataset(), batch_size=BATCH_SIZE)


def get_model():
    return ResNet18Proxy(num_classes=100)


def get_process_memory_mb():
    """Get current process RSS memory in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def benchmark_inference(model, loader):
    """Measure pure inference latency (batch_size=1)."""
    model.eval()
    with torch.no_grad():
        # Warmup
        for i, (x, _) in enumerate(loader):
            if i >= N_WARMUP:
                break
            _ = model(x)
        # Measure
        times = []
        for i, (x, _) in enumerate(loader):
            if i >= N_MEASURE:
                break
            t0 = time.perf_counter()
            _ = model(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms
    return np.mean(times), np.std(times)


def benchmark_training_step(model, optimizer, criterion, loader):
    """Measure one training step latency (batch_size=1). Slow on CPU so fewer steps."""
    model.train()
    times = []
    for i, (x, y) in enumerate(loader):
        if i >= 50:
            break
        x, y = x, y
        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        t1 = time.perf_counter()
        if i >= 5:  # skip first few
            times.append((t1 - t0) * 1000)
    return np.mean(times), np.std(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./dataset')
    parser.add_argument('--save_dir', default='.')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Device: {DEVICE}")
    loader = get_dataloader(args.data_dir)
    n_params = sum(p.numel() for p in get_model().parameters())

    results = {}
    configs = {
        'AdamW': lambda m: optim.AdamW(m.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=5e-4),
        'LAKTJU_NS': lambda m: __import__('sys').modules['optimizer.LAKTJU_NS'] if 'optimizer.LAKTJU_NS' in sys.modules else None,
        'LAKTJU_Lite': lambda m: None,
    }

    for name in ['AdamW', 'LAKTJU_NS', 'LAKTJU_Lite', 'MUON', 'SOAP']:
        print(f"\n=== {name} ===")
        torch.manual_seed(42)
        model = get_model()

        # Build optimizer
        if name == 'AdamW':
            opt = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=5e-4)
        elif name == 'LAKTJU_NS':
            from optimizer.LAKTJU_NS import LAKTJU_NS
            opt = LAKTJU_NS(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                            weight_decay=5e-4, ns_interval=500, ns_steps=1, ns_max_dim=256, min_ndim=2)
        elif name == 'LAKTJU_Lite':
            from optimizer.LAKTJU_Lite import LAKTJU_Lite
            opt = LAKTJU_Lite(model.parameters(), lr=1e-3, beta1=0.9, beta2=0.999,
                              weight_decay=5e-4, total_steps=1000)
        elif name in ('MUON', 'Muon'):
            from heavyball import Muon
            opt = Muon(list(model.parameters()), lr=1e-3, weight_decay=5e-4)
        elif name == 'SOAP':
            from heavyball import SOAP
            opt = SOAP(list(model.parameters()), lr=1e-3, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        # Model-only memory
        model_mem_mb = get_process_memory_mb()

        # Inference latency
        inf_ms, inf_std = benchmark_inference(model, loader)
        print(f"  Inference: {inf_ms:.2f} ± {inf_std:.2f} ms")

        # Training step latency
        train_ms, train_std = benchmark_training_step(model, opt, criterion, loader)
        print(f"  Train step: {train_ms:.2f} ± {train_std:.2f} ms")

        # Total memory (model + optimizer state)
        total_mem_mb = get_process_memory_mb()
        opt_mem_mb = total_mem_mb - model_mem_mb

        # Optimizer state size calculation
        opt_state_bytes = 0
        for group in opt.param_groups:
            for p in group['params']:
                state = opt.state.get(p, {})
                for v in state.values():
                    if isinstance(v, torch.Tensor):
                        opt_state_bytes += v.numel() * v.element_size()
        opt_state_mb = opt_state_bytes / 1024**2

        results[name] = {
            'inf_ms': float(inf_ms), 'inf_std': float(inf_std),
            'train_ms': float(train_ms), 'train_std': float(train_std),
            'model_mem_mb': float(model_mem_mb),
            'opt_state_mb': float(opt_state_mb),
            'total_mem_mb': float(total_mem_mb),
            'n_params': n_params,
        }
        print(f"  Model mem: {model_mem_mb:.0f} MB, Opt state: {opt_state_mb:.1f} MB, Total: {total_mem_mb:.0f} MB")
        del model, opt


    # Summary
    print(f"\n{'='*70}")
    print(f"CPU Edge Benchmark Summary (batch_size=1, CIFAR-100 ResNet-18)")
    print(f"{'='*70}")
    print(f"{'Method':<15} {'Inf(ms)':>9} {'Train(ms)':>10} {'Model(MB)':>9} {'Opt(MB)':>8} {'Total(MB)':>9}")
    print(f"{'-'*65}")
    adam_inf = results['AdamW']['inf_ms']
    for name in ['AdamW', 'LAKTJU_NS', 'LAKTJU_Lite']:
        r = results[name]
        slowdown = f"+{(r['inf_ms']/adam_inf - 1)*100:.1f}%"
        print(f"{name:<15} {r['inf_ms']:>8.2f} {r['train_ms']:>9.2f} "
              f"{r['model_mem_mb']:>9.0f} {r['opt_state_mb']:>8.1f} {r['total_mem_mb']:>9.0f}")

    os.makedirs('../results', exist_ok=True)
    with open('../results/cpu_edge_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to ../results/cpu_edge_benchmark.json")


if __name__ == '__main__':
    main()
