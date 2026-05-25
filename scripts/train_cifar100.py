"""
Self-contained CIFAR-100 training script for LafTJU-TII Phase 2.
Uses torchvision ResNet18. Supports all LafTJU optimizer variants.
"""
import os, sys, json, time, argparse, datetime, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_dataloaders(data_dir, batch_size=128, num_workers=4):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_ds = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
    test_ds = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)

    # 45K train, 5K val
    n_train = 45000
    n_val = 5000
    train_sub, val_sub = torch.utils.data.random_split(
        train_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def get_model():
    model = torchvision.models.resnet18(num_classes=100)
    # Replace first conv for CIFAR (32x32 images)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def build_optimizer(args, model, steps_per_epoch):
    params = model.parameters()
    opt_name = args.optimizer

    if opt_name == 'AdamW':
        lr = args.lr or 1e-3
        optimizer = optim.AdamW(params, lr=lr, betas=(0.9, 0.999),
                                weight_decay=args.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    elif opt_name == 'LAKTJU_NS':
        from optimizer.LAKTJU_NS import LAKTJU_NS
        lr = args.lr or 1e-3
        optimizer = LAKTJU_NS(params, lr=lr, betas=(0.9, 0.999),
                              weight_decay=args.weight_decay,
                              ns_interval=args.ns_interval, ns_steps=args.ns_steps,
                              ns_max_dim=args.ns_max_dim)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    elif opt_name == 'LAKTJU_Lite':
        from optimizer.LAKTJU_Lite import LAKTJU_Lite
        lr = args.lr or 5e-3
        total_steps = args.epochs * steps_per_epoch
        optimizer = LAKTJU_Lite(params, lr=lr, beta1=0.9, beta2=0.999,
                                weight_decay=args.weight_decay,
                                total_steps=total_steps)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    elif opt_name == 'LAKTJU':
        from optimizer.LAKTJU import LAKTJU
        lr = args.lr or 1e-3
        total_steps = args.epochs * steps_per_epoch
        optimizer = LAKTJU(params, tju_lr=lr * 10, a_lr=lr,
                           beta1=0.9, beta2=0.999,
                           weight_decay=args.weight_decay,
                           total_steps=total_steps,
                           kf_update_interval=20, kf_damping=1e-3,
                           homotopy_speed=8.0, warmup=100)
        # Register KF hooks on all Linear/Conv2d layers
        optimizer.register_hooks(model)
        scheduler = None  # LAKTJU has built-in LR schedule via homotopy

    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    return optimizer, scheduler


def train_epoch(model, loader, optimizer, criterion, opt_name):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (model(x).argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, optimizer=None, opt_name=None):
    model.eval()
    if optimizer is not None and opt_name == 'LAKTJU' and hasattr(optimizer, 'disable_kf_hooks'):
        optimizer.disable_kf_hooks()

    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    if optimizer is not None and opt_name == 'LAKTJU' and hasattr(optimizer, 'enable_kf_hooks'):
        optimizer.enable_kf_hooks()
    return total_loss / total, correct / total


def parse_args():
    p = argparse.ArgumentParser(description='CIFAR-100 LafTJU Training')
    p.add_argument('--optimizer', type=str, default='LAKTJU_NS',
                   choices=['AdamW', 'LAKTJU_NS', 'LAKTJU_Lite', 'LAKTJU'])
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data_dir', type=str, default='./dataset')
    p.add_argument('--save_dir', type=str, default='../results')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--ns_interval', type=int, default=500)
    p.add_argument('--ns_steps', type=int, default=1)
    p.add_argument('--ns_max_dim', type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_loader, val_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size, args.workers)
    steps_per_epoch = len(train_loader)

    model = get_model().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet18 (torchvision): {n_params:,} params")

    optimizer, scheduler = build_optimizer(args, model, steps_per_epoch)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.save_dir, exist_ok=True)
    tag = f"cifar100_resnet18_{args.optimizer}_seed{args.seed}"
    start = time.time()

    best_val_acc = 0
    best_test_acc = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, args.optimizer)

        val_loss, val_acc = evaluate(model, val_loader, criterion, optimizer, args.optimizer)
        test_loss, test_acc = evaluate(model, test_loader, criterion, optimizer, args.optimizer)

        if scheduler is not None:
            scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc,
            'test_loss': test_loss, 'test_acc': test_acc,
        })

        if epoch % 20 == 0 or epoch == args.epochs:
            elapsed = time.time() - start
            print(f"[{epoch:3d}/{args.epochs}] train_acc={train_acc:.4f} "
                  f"val_acc={val_acc:.4f} test_acc={test_acc:.4f} "
                  f"best={best_test_acc:.4f} ({elapsed:.0f}s)")

    total_time = time.time() - start

    result = {
        'config': vars(args),
        'best_val_acc': best_val_acc,
        'best_test_acc': best_test_acc,
        'final_test_acc': history[-1]['test_acc'],
        'history': history,
        'total_time': total_time,
        'n_params': n_params,
        'timestamp': datetime.datetime.now().isoformat(),
    }

    out_path = os.path.join(args.save_dir, f"{tag}.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=float)
    print(f"Results saved to {out_path}")
    print(f"Best test accuracy: {best_test_acc:.4f} ({best_test_acc*100:.2f}%)")


if __name__ == '__main__':
    main()
