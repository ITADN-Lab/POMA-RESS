"""
SWaT Industrial Water Treatment Anomaly Detection for LafTJU-TII.
Uses synthetic data matching the SWaT testbed structure (51 sensors/actuators).
Reference: Goh et al. (2016) "A Dataset to Support Research in the Design
of Secure Water Treatment Systems", iTrust, SUTD.
"""
import os, sys, json, time, argparse
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SWaT process: 51 sensors (25 sensors + 26 actuators)
SWAT_CONFIG = {
    'n_sensors': 25, 'n_actuators': 26, 'n_total': 51,
    'normal_duration': 7 * 24 * 60,  # 7 days, per-minute (10080 samples)
    'attack_scenarios': 36,
    'description': 'Secure Water Treatment (SWaT) testbed'
}

def generate_swat_data(n_samples=10000, n_faults=6, seed=42, correlation=0.85):
    """Generate synthetic SWaT-like data with normal and attack conditions.

    Process model: 51 variables with multi-stage water treatment dynamics.
    Attack types: single-point, multi-point, scaling, ramp, pulse.
    """
    rng = np.random.default_rng(seed)
    n_feats = 51

    # Normal operation: correlated multivariate time series
    base = rng.normal(0, 1, (n_samples, n_feats))
    # Add temporal correlation via recursive smoothing
    for i in range(1, n_samples):
        base[i] = correlation * base[i-1] + (1-correlation) * base[i]
    # Add cross-variable correlation (process coupling)
    coupling = rng.normal(0, 0.3, (n_feats, n_feats))
    np.fill_diagonal(coupling, 1.0)
    data_normal = base @ coupling.T

    # Generate attack data for each fault type
    samples_per_fault = n_samples // n_faults
    data_faults = []
    labels_faults = []

    fault_types = [
        ('single_point_bias', lambda x: x + rng.choice([-5, 5], size=x.shape)),
        ('multi_point_drift', lambda x: x + np.cumsum(rng.normal(0, 0.1, x.shape), axis=0)),
        ('scaling_attack', lambda x: x * rng.uniform(0.1, 3.0, size=x.shape)),
        ('ramp_attack', lambda x: x + np.linspace(0, 8, len(x))[:, None]),
        ('pulse_attack', lambda x: x + 8 * (rng.random(x.shape) < 0.1)),
        ('random_variation', lambda x: x * (1 + rng.normal(0, 0.5, x.shape))),
    ]

    for fid, (fname, fault_fn) in enumerate(fault_types):
        # Start from normal data, then apply fault
        fault_data = data_normal[rng.choice(n_samples, samples_per_fault, replace=False)].copy()
        fault_data = fault_fn(fault_data)
        data_faults.append(fault_data)
        labels_faults.extend([fid + 1] * samples_per_fault)  # 0=normal, 1+=fault

    # Combine
    X = np.vstack([data_normal[:n_samples // 2], np.vstack(data_faults)])
    y = np.array([0] * (n_samples // 2) + labels_faults)

    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


class MLPClassifier(nn.Module):
    """3-layer MLP for industrial anomaly detection."""
    def __init__(self, input_dim, hidden=256, num_classes=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden//2, num_classes),
        )
    def forward(self, x): return self.net(x)


def build_optimizer(args, model, steps_per_epoch):
    params = model.parameters(); opt = args.optimizer
    if opt == 'AdamW':
        optimizer = optim.AdamW(params, lr=args.lr or 1e-3, weight_decay=args.weight_decay)
    elif opt == 'LAKTJU_NS':
        from optimizer.LAKTJU_NS import LAKTJU_NS
        optimizer = LAKTJU_NS(params, lr=args.lr or 1e-3, betas=(0.9,0.999),
                              weight_decay=args.weight_decay, ns_interval=args.ns_interval,
                              ns_steps=args.ns_steps, ns_max_dim=args.ns_max_dim, min_ndim=2)
    elif opt == 'LAKTJU_Lite':
        from optimizer.LAKTJU_Lite import LAKTJU_Lite
        optimizer = LAKTJU_Lite(params, lr=args.lr or 1e-3, beta1=0.9, beta2=0.999,
                                weight_decay=args.weight_decay, total_steps=args.epochs*steps_per_epoch)
    elif opt in ('MUON', 'Muon'):
        from heavyball import Muon
        optimizer = Muon(list(params), lr=args.lr or 1e-3, weight_decay=args.weight_decay)
    elif opt == 'SOAP':
        from heavyball import SOAP
        optimizer = SOAP(list(params), lr=args.lr or 1e-3, weight_decay=args.weight_decay)
    elif opt == 'Adan':
        from adan_pytorch import Adan
        optimizer = Adan(list(params), lr=args.lr or 1e-3, weight_decay=args.weight_decay,
                         betas=(0.98, 0.92, 0.99))
    else: raise ValueError(opt)
    return optimizer, lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


def train_epoch(model, loader, opt, criterion):
    model.train(); total_loss, correct, total = 0,0,0
    for x,y in loader:
        x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad()
        loss=criterion(model(x),y); loss.backward(); opt.step()
        total_loss+=loss.item()*x.size(0)
        correct+=(model(x).argmax(1)==y).sum().item(); total+=x.size(0)
    return total_loss/total, correct/total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval(); total_loss, correct, total = 0,0,0
    for x,y in loader:
        x,y=x.to(DEVICE),y.to(DEVICE)
        loss=criterion(model(x),y); total_loss+=loss.item()*x.size(0)
        correct+=(model(x).argmax(1)==y).sum().item(); total+=x.size(0)
    return total_loss/total, correct/total

def parse_args():
    p = argparse.ArgumentParser(description='SWaT Industrial Anomaly Detection')
    p.add_argument('--optimizer', default='LAKTJU_NS', choices=['AdamW','LAKTJU_NS','LAKTJU_Lite'])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n_samples', type=int, default=20000)
    p.add_argument('--correlation', type=float, default=0.85)
    p.add_argument('--save_dir', default='./results')
    p.add_argument('--ns_interval', type=int, default=50)
    p.add_argument('--ns_steps', type=int, default=1)
    p.add_argument('--ns_max_dim', type=int, default=256)
    return p.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    X, y = generate_swat_data(args.n_samples, correlation=args.correlation, seed=args.seed)
    scaler = StandardScaler(); X = scaler.fit_transform(X)

    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.15, random_state=args.seed, stratify=y_tr)

    tr_ds = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
    va_ds = TensorDataset(torch.FloatTensor(X_va), torch.LongTensor(y_va))
    ts_ds = TensorDataset(torch.FloatTensor(X_ts), torch.LongTensor(y_ts))
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size)
    ts_ld = DataLoader(ts_ds, batch_size=args.batch_size)

    n_classes = len(np.unique(y))
    model = MLPClassifier(51, num_classes=n_classes).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SWaT-Synth: {args.n_samples} samples, {51} vars, {n_classes} classes, {n_params:,} params, corr={args.correlation}")

    optimizer, scheduler = build_optimizer(args, model, len(tr_ld))
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.save_dir, exist_ok=True)
    tag = f"swat_synth_{args.optimizer}_corr{args.correlation}_seed{args.seed}"
    best_val, best_test, history = 0, 0, []; start = time.time()

    for epoch in range(1, args.epochs+1):
        tl,ta = train_epoch(model, tr_ld, optimizer, criterion)
        vl,va = evaluate(model, va_ld, criterion)
        sl,sa = evaluate(model, ts_ld, criterion)
        scheduler.step()
        if va > best_val: best_val = va; best_test = sa
        history.append({'epoch':epoch,'train_acc':ta,'val_acc':va,'test_acc':sa})
        if epoch % 20 == 0:
            print(f"[{epoch:3d}/{args.epochs}] train={ta:.3f} val={va:.3f} test={sa:.3f} best={best_test:.3f}")

    elapsed = time.time()-start
    result = {'config':vars(args),'best_val_acc':best_val,'best_test_acc':best_test,
              'history':history,'total_time':elapsed,'n_params':n_params}
    fp = os.path.join(args.save_dir, f"{tag}.json")
    with open(fp,'w') as f: json.dump(result, f, indent=2, default=float)
    print(f"Saved: {fp} | Best test: {best_test:.3f}")

if __name__=='__main__': main()
