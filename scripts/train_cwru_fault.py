"""
CWRU Bearing Fault Diagnosis Experiment for LafTJU-TII Paper.

Dataset: Case Western Reserve University Bearing Data Center
  - 10 classes: Normal + 9 fault types (ball/inner/outer race, 0.007/0.014/0.021 inch)
  - Drive end accelerometer at 12 kHz
  - Sliding window segmentation with 50% overlap

Model: 1D-ResNet (5-layer) for vibration signal classification
Optimizers: AdamW, LAKTJU_NS, LAKTJU_Lite, SGD
"""
import os
import sys
import math
import json
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ────────────────────────────────────────────────────────────────
# Data: CWRU Bearing Dataset
# ────────────────────────────────────────────────────────────────

FAULT_LABELS = {
    'Normal': 0,
    'Ball_007': 1, 'Ball_014': 2, 'Ball_021': 3,
    'Inner_007': 4, 'Inner_014': 5, 'Inner_021': 6,
    'Outer_007': 7, 'Outer_014': 8, 'Outer_021': 9,
}
NUM_CLASSES = 10


def load_mat_file(filepath):
    """Load a single .mat file and extract drive end vibration data."""
    from scipy.io import loadmat
    mat = loadmat(filepath)
    # Drive end accelerometer data key varies; try common patterns
    for key in mat:
        if 'DE' in key or 'drive' in key.lower():
            return mat[key].flatten()
    # Fallback: first non-header array
    for key in mat:
        if not key.startswith('_'):
            return mat[key].flatten()
    raise ValueError(f"No vibration data found in {filepath}")


def segment_signal(signal, window_size=2048, overlap=0.5):
    """Segment a vibration signal into overlapping windows."""
    step = int(window_size * (1 - overlap))
    segments = []
    for start in range(0, len(signal) - window_size + 1, step):
        segments.append(signal[start:start + window_size])
    return np.array(segments, dtype=np.float32)


class CWRUDataset(Dataset):
    """CWRU Bearing Fault Diagnosis Dataset.

    Expected data directory structure:
        data_dir/
        ├── Normal/
        │   ├── 97.mat  (0 hp)
        │   └── ...
        ├── Ball_007/
        │   ├── 118.mat
        │   └── ...
        ├── Inner_007/
        │   ├── 105.mat
        │   └── ...
        └── Outer_007/
            ├── ...
    """
    def __init__(self, data_dir, window_size=2048, overlap=0.5,
                 max_samples_per_class=200, normalize=True):
        self.window_size = window_size
        self.samples = []
        self.labels = []

        for fault_name, label in FAULT_LABELS.items():
            fault_dir = os.path.join(data_dir, fault_name)
            if not os.path.isdir(fault_dir):
                print(f"Warning: {fault_dir} not found, skipping")
                continue

            all_segments = []
            for fname in sorted(os.listdir(fault_dir)):
                if fname.endswith('.mat'):
                    fpath = os.path.join(fault_dir, fname)
                    signal = load_mat_file(fpath)
                    segs = segment_signal(signal, window_size, overlap)
                    all_segments.append(segs)

            if not all_segments:
                continue
            all_segments = np.concatenate(all_segments, axis=0)

            # Subsample if too many
            if len(all_segments) > max_samples_per_class:
                idx = np.random.choice(len(all_segments), max_samples_per_class, replace=False)
                all_segments = all_segments[idx]

            self.samples.append(all_segments)
            self.labels.extend([label] * len(all_segments))

        self.samples = np.concatenate(self.samples, axis=0)
        self.labels = np.array(self.labels, dtype=np.int64)

        # Normalize: zero mean, unit std per sample
        if normalize:
            mean = self.samples.mean(axis=1, keepdims=True)
            std = self.samples.std(axis=1, keepdims=True) + 1e-8
            self.samples = (self.samples - mean) / std

        print(f"CWRU Dataset: {len(self.samples)} samples, {NUM_CLASSES} classes, "
              f"window_size={window_size}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32).unsqueeze(0)  # (1, W)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ────────────────────────────────────────────────────────────────
# Model: 1D-ResNet
# ────────────────────────────────────────────────────────────────

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7,
                               stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7,
                               stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet1D(nn.Module):
    """1D ResNet for vibration signal classification.

    Architecture:
      conv1 (1→64, k=15, s=2) → bn → relu → maxpool
      layer1: 2 × BasicBlock1D(64)
      layer2: 2 × BasicBlock1D(64→128, s=2)
      layer3: 2 × BasicBlock1D(128→256, s=2)
      layer4: 2 × BasicBlock1D(256→512, s=2)
      avgpool → fc(512 → num_classes)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        layers = [BasicBlock1D(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


# ────────────────────────────────────────────────────────────────
# Optimizer factory
# ────────────────────────────────────────────────────────────────

def build_optimizer(args, model):
    params = model.parameters()
    opt_name = args.optimizer

    if opt_name == 'SGD':
        lr = args.lr or 0.01
        optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    elif opt_name == 'Adam':
        lr = args.lr or 1e-3
        optimizer = optim.Adam(params, lr=lr, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    elif opt_name == 'AdamW':
        lr = args.lr or 1e-3
        optimizer = optim.AdamW(params, lr=lr, betas=(0.9, 0.999),
                                weight_decay=args.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    elif opt_name == 'LAKTJU_NS':
        from optimizer.LAKTJU_NS import LAKTJU_NS
        lr = args.lr or 5e-3
        optimizer = LAKTJU_NS(params, lr=lr, betas=(0.9, 0.999),
                              weight_decay=args.weight_decay,
                              ns_interval=args.ns_interval, ns_steps=args.ns_steps,
                              ns_max_dim=args.ns_max_dim)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    elif opt_name == 'LAKTJU_NS_Adam':
        from optimizer.LAKTJU_NS_Adam import LAKTJU_NS_Adam
        lr = args.lr or 1e-3
        optimizer = LAKTJU_NS_Adam(params, lr=lr, betas=(0.9, 0.999),
                                   weight_decay=args.weight_decay,
                                   ns_interval=args.ns_interval, ns_steps=args.ns_steps,
                                   ns_max_dim=args.ns_max_dim)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    elif opt_name == 'LAKTJU_Lite':
        from optimizer.LAKTJU_Lite import LAKTJU_Lite
        lr = args.lr or 5e-3
        optimizer = LAKTJU_Lite(params, lr=lr, beta1=0.9, beta2=0.999,
                                weight_decay=args.weight_decay,
                                total_steps=args.epochs * 100)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    elif opt_name == 'LAKTJU':
        from optimizer.LAKTJU import LAKTJU
        lr = args.lr or 1e-3
        optimizer = LAKTJU(params, tju_lr=lr * 10, a_lr=lr,
                           total_steps=args.epochs * 100)
        # LAKTJU uses tju_lr/a_lr in param groups; wrap scheduler
        scheduler = None
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    return optimizer, scheduler


# ────────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, total_correct / total


def parse_args():
    p = argparse.ArgumentParser(description='CWRU Fault Diagnosis — LafTJU-TII')
    p.add_argument('--data_dir', type=str, default='../data/cwru')
    p.add_argument('--optimizer', type=str, default='AdamW',
                   choices=['SGD', 'Adam', 'AdamW', 'LAKTJU_NS', 'LAKTJU_NS_Adam',
                            'LAKTJU_Lite', 'LAKTJU'])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--window_size', type=int, default=2048)
    p.add_argument('--max_samples_per_class', type=int, default=200)
    p.add_argument('--save_dir', type=str, default='../results')
    p.add_argument('--ns_interval', type=int, default=100)
    p.add_argument('--ns_steps', type=int, default=1)
    p.add_argument('--ns_max_dim', type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    dataset = CWRUDataset(args.data_dir, window_size=args.window_size,
                          max_samples_per_class=args.max_samples_per_class)

    # 70/15/15 split
    n = len(dataset)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    model = ResNet1D(num_classes=NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet1D: {n_params:,} parameters")

    # Optimizer
    optimizer, scheduler = build_optimizer(args, model)

    # Register KF hooks for LAKTJU (full) optimizer
    if args.optimizer == 'LAKTJU' and hasattr(optimizer, 'register_hooks'):
        optimizer.register_hooks(model)

    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = 0
    best_test_acc = 0
    history = []

    os.makedirs(args.save_dir, exist_ok=True)
    tag = f"cwru_{args.optimizer}_seed{args.seed}"
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Disable KF hooks during evaluation (LAKTJU full)
        if args.optimizer == 'LAKTJU' and hasattr(optimizer, 'disable_kf_hooks'):
            optimizer.disable_kf_hooks()

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        if args.optimizer == 'LAKTJU' and hasattr(optimizer, 'enable_kf_hooks'):
            optimizer.enable_kf_hooks()

        if scheduler is not None:
            scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        history.append({
            'epoch': epoch,
            'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc,
            'test_loss': test_loss, 'test_acc': test_acc,
            'best_val_acc': best_val_acc, 'best_test_acc': best_test_acc,
        })

        if epoch % 10 == 0 or epoch == args.epochs:
            elapsed = time.time() - start
            print(f"[{epoch:3d}/{args.epochs}] train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
                  f"test_acc={test_acc:.4f} best={best_test_acc:.4f} ({elapsed:.0f}s)")

    total_time = time.time() - start

    # Save results
    result = {
        'config': vars(args),
        'final_test_acc': history[-1]['test_acc'],
        'best_test_acc': best_test_acc,
        'best_val_acc': best_val_acc,
        'history': history,
        'total_time': total_time,
        'n_params': n_params,
        'timestamp': datetime.datetime.now().isoformat(),
    }

    out_path = os.path.join(args.save_dir, f"{tag}.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {out_path}")
    print(f"Best test accuracy: {best_test_acc:.4f}")


if __name__ == '__main__':
    main()
