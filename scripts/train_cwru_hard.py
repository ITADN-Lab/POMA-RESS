"""
CWRU-Hard: Industrial robustness benchmark for LafTJU-TII.
Adds realistic industrial perturbations to CWRU bearing data:
- Gaussian sensor noise (SNR 10-20 dB)
- Random sensor dropout (10-30% of time steps zeroed)
- Label noise (5-10% labels flipped)
Tests optimizer robustness to industrial data quality issues.
"""
import os, sys, json, time, argparse, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.io import loadmat
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FAULT_LABELS = {'Normal':0,'Ball_007':1,'Ball_014':2,'Ball_021':3,
    'Inner_007':4,'Inner_014':5,'Inner_021':6,'Outer_007':7,'Outer_014':8,'Outer_021':9}

def load_mat_file(fp):
    mat = loadmat(fp)
    for key in mat:
        if 'DE' in key or 'drive' in key.lower(): return mat[key].flatten()
    for key in mat:
        if not key.startswith('_'): return mat[key].flatten()
    raise ValueError(f"No data in {fp}")

def segment_signal(signal, window_size=2048, overlap=0.5):
    step = int(window_size * (1 - overlap))
    segs = [signal[s:s+window_size] for s in range(0, len(signal)-window_size+1, step)]
    return np.array(segs, dtype=np.float32)

class CWRUHardDataset(Dataset):
    """CWRU with industrial perturbations."""
    def __init__(self, data_dir, window_size=2048, overlap=0.5,
                 max_per_class=200, noise_snr_db=15, dropout_pct=0.15, label_noise=0.05, seed=42):
        rng = np.random.default_rng(seed)
        self.samples, self.labels = [], []

        for fault_name, label in FAULT_LABELS.items():
            fault_dir = os.path.join(data_dir, fault_name)
            if not os.path.isdir(fault_dir): continue
            all_segs = []
            for fname in sorted(os.listdir(fault_dir)):
                if fname.endswith('.mat'):
                    signal = load_mat_file(os.path.join(fault_dir, fname))
                    all_segs.append(segment_signal(signal, window_size, overlap))
            if not all_segs: continue
            all_segs = np.concatenate(all_segs, axis=0)
            if len(all_segs) > max_per_class:
                idx = rng.choice(len(all_segs), max_per_class, replace=False)
                all_segs = all_segs[idx]
            self.samples.append(all_segs)
            self.labels.extend([label] * len(all_segs))

        self.samples = np.concatenate(self.samples, axis=0)
        self.labels = np.array(self.labels, dtype=np.int64)

        # Apply perturbations
        for i in range(len(self.samples)):
            sig = self.samples[i]
            # Gaussian noise
            sig_power = np.mean(sig**2)
            noise_power = sig_power / (10**(noise_snr_db/10))
            sig += rng.normal(0, np.sqrt(noise_power), sig.shape).astype(np.float32)
            # Random dropout
            drop_mask = rng.random(len(sig)) < dropout_pct
            sig[drop_mask] = 0
            # Label noise
            if rng.random() < label_noise:
                other_labels = [l for l in range(10) if l != self.labels[i]]
                self.labels[i] = int(rng.choice(other_labels))
            self.samples[i] = sig

        # Normalize
        mean = self.samples.mean(axis=1, keepdims=True)
        std = self.samples.std(axis=1, keepdims=True) + 1e-8
        self.samples = (self.samples - mean) / std

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        return (torch.tensor(self.samples[idx], dtype=torch.float32).unsqueeze(0),
                torch.tensor(self.labels[idx], dtype=torch.long))


class ResNet1D(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_c, out_c, blocks, stride=1):
        layers = []
        layers.append(nn.Conv1d(in_c, out_c, 3, stride, 1, bias=False))
        layers.append(nn.BatchNorm1d(out_c)); layers.append(nn.ReLU(inplace=True))
        for _ in range(1, blocks):
            layers.append(nn.Conv1d(out_c, out_c, 3, 1, 1, bias=False))
            layers.append(nn.BatchNorm1d(out_c)); layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


def build_optimizer(args, model):
    params = model.parameters()
    opt = args.optimizer
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
                                weight_decay=args.weight_decay, total_steps=args.epochs*100)
    elif opt == 'LAKTJU':
        from optimizer.LAKTJU import LAKTJU
        lr = args.lr or 1e-3
        optimizer = LAKTJU(params, tju_lr=lr*10, a_lr=lr, weight_decay=args.weight_decay,
                           total_steps=args.epochs*100)
        optimizer.register_hooks(model)
        return optimizer, None
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
        loss=criterion(model(x),y)
        total_loss+=loss.item()*x.size(0)
        correct+=(model(x).argmax(1)==y).sum().item(); total+=x.size(0)
    return total_loss/total, correct/total

def parse_args():
    p = argparse.ArgumentParser(description='CWRU-Hard Industrial Robustness')
    p.add_argument('--optimizer', default='LAKTJU_NS', choices=['AdamW','LAKTJU_NS','LAKTJU_Lite','LAKTJU'])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data_dir', default='experiments/data/cwru')
    p.add_argument('--save_dir', default='./results')
    p.add_argument('--ns_interval', type=int, default=50)
    p.add_argument('--ns_steps', type=int, default=1)
    p.add_argument('--ns_max_dim', type=int, default=256)
    p.add_argument('--noise_snr', type=float, default=10)  # lower = more noise
    p.add_argument('--dropout_pct', type=float, default=0.2)
    p.add_argument('--label_noise', type=float, default=0.08)
    return p.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    ds = CWRUHardDataset(args.data_dir, noise_snr_db=args.noise_snr,
                         dropout_pct=args.dropout_pct, label_noise=args.label_noise, seed=args.seed)
    n = len(ds); n_tr = int(0.7*n); n_val = int(0.15*n); n_ts = n-n_tr-n_val
    tr,va,te = random_split(ds, [n_tr,n_val,n_ts], generator=torch.Generator().manual_seed(args.seed))
    tr_ld = DataLoader(tr, batch_size=args.batch_size, shuffle=True)
    va_ld = DataLoader(va, batch_size=args.batch_size)
    te_ld = DataLoader(te, batch_size=args.batch_size)

    model = ResNet1D().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CWRU-Hard: {len(ds)} samples, SNR={args.noise_snr}dB, dropout={args.dropout_pct}, label_noise={args.label_noise}")
    print(f"Model: {n_params:,} params")

    optimizer, scheduler = build_optimizer(args, model)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.save_dir, exist_ok=True)
    tag = f"cwru_hard_{args.optimizer}_snr{args.noise_snr}_seed{args.seed}"
    best_val, best_test, history = 0, 0, []; start = time.time()

    for epoch in range(1, args.epochs+1):
        tl,ta = train_epoch(model, tr_ld, optimizer, criterion)
        vl,va_c = evaluate(model, va_ld, criterion)
        sl,sa = evaluate(model, te_ld, criterion)
        if scheduler: scheduler.step()
        if va_c > best_val: best_val = va_c; best_test = sa
        history.append({'epoch':epoch,'train_acc':ta,'val_acc':va_c,'test_acc':sa})
        if epoch % 20 == 0:
            print(f"[{epoch:3d}/{args.epochs}] train={ta:.3f} val={va_c:.3f} test={sa:.3f} best={best_test:.3f}")

    elapsed = time.time()-start
    result = {'config':vars(args),'best_val_acc':best_val,'best_test_acc':best_test,
              'history':history,'total_time':elapsed,'n_params':n_params}
    fp = os.path.join(args.save_dir, f"{tag}.json")
    with open(fp,'w') as f: json.dump(result, f, indent=2, default=float)
    print(f"Saved: {fp} | Best test: {best_test:.3f}")

if __name__=='__main__': main()
