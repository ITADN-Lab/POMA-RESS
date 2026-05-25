"""
SECOM Semiconductor Manufacturing Fault Detection for LafTJU-TII.
Dataset: UCI SECOM — 1567 samples, 591 features, binary classification (pass/fail).
Industrial relevance: semiconductor manufacturing process monitoring.
"""
import os, sys, json, time, argparse, urllib.request, io, zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_secom(data_dir='./data'):
    """Download and load the SECOM dataset."""
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, 'secom.data')
    labels_path = os.path.join(data_dir, 'secom_labels.data')

    if not os.path.exists(data_path):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data'
        urllib.request.urlretrieve(url, data_path)
        url_l = 'https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data'
        urllib.request.urlretrieve(url_l, labels_path)

    X = np.loadtxt(data_path)
    # Labels format: "label \"date time\"" — parse first token as label
    raw_labels = []
    with open(labels_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            label_str = line.split()[0]  # first token is the label
            try:
                raw_labels.append(int(label_str))
            except ValueError:
                continue
    y = np.array(raw_labels)
    y[y == -1] = 0  # -1 = pass, 1 = fail -> 0 = pass, 1 = fail
    return X, y


class MLP(nn.Module):
    """3-layer MLP for SECOM classification."""
    def __init__(self, input_dim, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x): return self.net(x).squeeze(-1)


def build_optimizer(args, model, steps_per_epoch):
    params = model.parameters()
    opt = args.optimizer
    if opt == 'AdamW':
        optimizer = optim.AdamW(params, lr=args.lr or 1e-3, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif opt == 'LAKTJU_NS':
        from optimizer.LAKTJU_NS import LAKTJU_NS
        optimizer = LAKTJU_NS(params, lr=args.lr or 1e-3, betas=(0.9, 0.999),
                              weight_decay=args.weight_decay,
                              ns_interval=args.ns_interval, ns_steps=args.ns_steps,
                              ns_max_dim=args.ns_max_dim, min_ndim=2)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif opt == 'LAKTJU_Lite':
        from optimizer.LAKTJU_Lite import LAKTJU_Lite
        total_steps = args.epochs * steps_per_epoch
        optimizer = LAKTJU_Lite(params, lr=args.lr or 1e-3, beta1=0.9, beta2=0.999,
                                weight_decay=args.weight_decay, total_steps=total_steps)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif opt == 'LAKTJU':
        from optimizer.LAKTJU import LAKTJU
        optimizer = LAKTJU(params, tju_lr=(args.lr or 1e-3) * 10, a_lr=args.lr or 1e-3,
                           weight_decay=args.weight_decay, total_steps=args.epochs * steps_per_epoch)
        optimizer.register_hooks(model)
        scheduler = None
    elif opt in ('MUON', 'Muon'):
        from heavyball import Muon
        optimizer = Muon(list(params), lr=args.lr or 1e-3, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif opt == 'SOAP':
        from heavyball import SOAP
        optimizer = SOAP(list(params), lr=args.lr or 1e-3, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif opt == 'Adan':
        from adan_pytorch import Adan
        optimizer = Adan(list(params), lr=args.lr or 1e-3, weight_decay=args.weight_decay,
                         betas=(0.98, 0.92, 0.99))
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise ValueError(opt)
    return optimizer, scheduler


def train_epoch(model, loader, optimizer, criterion, opt_name):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        total_loss += criterion(logits, y.float()).item() * x.size(0)
        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def parse_args():
    p = argparse.ArgumentParser(description='SECOM Industrial Fault Detection')
    p.add_argument('--optimizer', default='LAKTJU_NS', choices=['AdamW', 'LAKTJU_NS', 'LAKTJU_Lite', 'LAKTJU', 'MUON', 'Muon', 'SOAP', 'Adan'])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ns_interval', type=int, default=50)
    p.add_argument('--ns_steps', type=int, default=1)
    p.add_argument('--ns_max_dim', type=int, default=256)
    p.add_argument('--save_dir', default='./results')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    X, y = load_secom()
    # Preprocessing
    imp = SimpleImputer(strategy='mean')
    X = imp.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=args.seed, stratify=y_train)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = MLP(X.shape[1]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SECOM: {X.shape[1]} features, {n_params:,} params")

    optimizer, scheduler = build_optimizer(args, model, len(train_loader))
    # Weighted loss for imbalance
    pos_weight = torch.tensor([(y_train == 0).sum() / max(y_train.sum(), 1)]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    os.makedirs(args.save_dir, exist_ok=True)
    tag = f"secom_{args.optimizer}_seed{args.seed}"
    best_val, best_test = 0, 0
    start = time.time()
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, args.optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        if scheduler: scheduler.step()

        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc

        history.append({'epoch': epoch, 'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc})
        if epoch % 20 == 0 or epoch == args.epochs:
            print(f"[{epoch:3d}/{args.epochs}] train_acc={train_acc:.3f} val_acc={val_acc:.3f} test_acc={test_acc:.3f} best={best_test:.3f}")

    total_time = time.time() - start
    result = {'config': vars(args), 'best_val_acc': best_val, 'best_test_acc': best_test,
              'history': history, 'total_time': total_time, 'n_params': n_params,
              'input_dim': X.shape[1]}
    out_path = os.path.join(args.save_dir, f"{tag}.json")
    with open(out_path, 'w') as f: json.dump(result, f, indent=2, default=float)
    print(f"Saved to {out_path} | Best test: {best_test:.3f}")

if __name__ == '__main__':
    main()
