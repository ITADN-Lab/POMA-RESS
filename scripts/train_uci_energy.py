"""UCI Appliances Energy Prediction — Industrial building energy time series.

Dataset: https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
19735 samples, 28 features, continuous energy consumption regression.
Temporal structure from building sensor readings (temperature, humidity, etc.).

Downloaded automatically from UCI repository when not found locally.
"""
import os, sys, json, time, argparse, datetime, urllib.request, zipfile
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, '..', '..'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

UCI_URL = 'https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip'
LOCAL_ZIP = os.path.join(THIS_DIR, '..', 'data', 'appliances_energy.zip')
LOCAL_CSV = os.path.join(THIS_DIR, '..', 'data', 'energydata_complete.csv')


class MLP(nn.Module):
    def __init__(self, input_dim, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden=64, num_layers=2, window=10, dropout=0.2):
        super().__init__()
        self.window = window
        self.input_dim = input_dim
        self.lstm = nn.LSTM(input_dim, hidden, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))
    def forward(self, x):
        B, T, D = x.shape
        # x: (B, T, D) — already batched windows from Dataset
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze(-1)


def download():
    os.makedirs(os.path.dirname(LOCAL_ZIP), exist_ok=True)
    if not os.path.exists(LOCAL_CSV):
        print(f"Downloading from {UCI_URL}...")
        urllib.request.urlretrieve(UCI_URL, LOCAL_ZIP)
        with zipfile.ZipFile(LOCAL_ZIP, 'r') as zf:
            zf.extractall(os.path.dirname(LOCAL_ZIP))
        os.remove(LOCAL_ZIP)
        print(f"Extracted to {LOCAL_CSV}")


def load_data(window=10, test_size=0.2, val_size=0.15):
    download()
    import pandas as pd
    df = pd.read_csv(LOCAL_CSV)
    # Target is Appliances energy (first column after date)
    y = df['Appliances'].values.astype(np.float32)
    # Use all numeric columns except date, lights (unreliable), and appliance
    drop_cols = ['date', 'Appliances', 'lights']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].values.astype(np.float32)

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # For LSTM: create sequences
    if window > 1:
        X_seq, y_seq = [], []
        for i in range(len(X) - window):
            X_seq.append(X[i:i+window])
            y_seq.append(y[i+window])
        X, y = np.array(X_seq), np.array(y_seq)
    return X, y


def build_optimizer(name, params, lr, wd):
    if name == 'AdamW':
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif name == 'LAKTJU_NS':
        sys.path.insert(0, os.path.join(THIS_DIR, '..', '..'))
        from optimizer.LAKTJU_NS import LAKTJU_NS
        return LAKTJU_NS(params, lr=lr, betas=(0.9, 0.999), weight_decay=wd,
                         ns_interval=50, ns_steps=1, ns_max_dim=256, min_ndim=2)
    elif name in ('MUON', 'Muon'):
        from heavyball import Muon
        return Muon(list(params), lr=lr, weight_decay=wd)
    elif name == 'SOAP':
        from heavyball import SOAP
        return SOAP(list(params), lr=lr, weight_decay=wd)
    else:
        raise ValueError(name)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--optimizer', default='AdamW', choices=['AdamW','LAKTJU_NS','MUON','SOAP'])
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--model', default='lstm', choices=['lstm', 'mlp'])
    p.add_argument('--window', type=int, default=10)
    p.add_argument('--save_dir', required=True)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    print("Loading Appliances Energy dataset...")
    X, y = load_data(window=args.window if args.model == 'lstm' else 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=args.seed)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, args.batch_size)
    test_loader = DataLoader(test_ds, args.batch_size)

    if args.model == 'lstm':
        model = LSTM(X.shape[-1], window=args.window).to(DEVICE)
    else:
        model = MLP(X.shape[1]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
    criterion = nn.MSELoss()

    os.makedirs(args.save_dir, exist_ok=True)
    tag = f"energy_{args.model}_{args.optimizer}_seed{args.seed}"
    out_path = os.path.join(args.save_dir, f"{tag}.json")

    best_val_rmse, best_test_rmse = float('inf'), float('inf')
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total = 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0); total += x.size(0)

        # Evaluate
        model.eval()
        with torch.no_grad():
            val_preds, val_targets = [], []
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                val_preds.append(model(x).cpu().numpy())
                val_targets.append(y.cpu().numpy())
        val_rmse = float(np.sqrt(np.mean((np.concatenate(val_preds) - np.concatenate(val_targets))**2)))
        scheduler.step(val_rmse)

        with torch.no_grad():
            test_preds, test_targets = [], []
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                test_preds.append(model(x).cpu().numpy())
                test_targets.append(y.cpu().numpy())
        test_rmse = float(np.sqrt(np.mean((np.concatenate(test_preds) - np.concatenate(test_targets))**2)))

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse; best_test_rmse = test_rmse

        if epoch % 20 == 0 or epoch == args.epochs:
            print(f"[{epoch:3d}/{args.epochs}] val_rmse={val_rmse:.3f} test_rmse={test_rmse:.3f} best={best_test_rmse:.3f}")

    result = {
        'config': vars(args), 'n_params': n_params,
        'best_test_rmse': float(best_test_rmse), 'best_val_rmse': float(best_val_rmse),
        'total_time': time.time() - t0,
    }
    with open(out_path, 'w') as f: json.dump(result, f, indent=2, default=float)
    print(f"Saved {out_path} | best_test_rmse={best_test_rmse:.3f}")


if __name__ == '__main__':
    main()
