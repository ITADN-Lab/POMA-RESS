"""
XJTU-SY Bearing RUL training (LSTM) — mirrors train_cmapss_rul.py interface.

Cross-condition evaluation protocol (recommended for TII industrial validation):
  --train_oc OC1,OC2 --test_oc OC3   # train on two operating conditions, test on held-out
Or single-condition random-split:
  --train_oc OC1 --test_oc OC1 --random_split_within_oc

Outputs: experiments/results_xjtu/xjtu_<train>_to_<test>_<opt>_seed<seed>_<tag>.json
"""
import os, sys, json, time, argparse, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, '..', '..'))

# Reuse optimizer builders from C-MAPSS training script (import the function)
sys.path.insert(0, THIS_DIR)
from train_cmapss_rul import build_optimizer as _build_optimizer_from_cmapss  # type: ignore


def kappa_of(M):
    """Condition number sigma_max/sigma_min of a (reshaped-2D) momentum buffer."""
    m2d = M.detach().reshape(M.shape[0], -1).double()
    s = torch.linalg.svdvals(m2d)
    s = s[s > 1e-10]
    if s.numel() < 2:
        return None
    return float(s[0] / s[-1])


class XJTUDataset(Dataset):
    def __init__(self, npz_paths, window=10, normalize_stats=None):
        self.window = window
        self.segments = []   # list of (features, rul, bearing_idx)
        all_feats = []
        for path in npz_paths:
            d = np.load(path)
            feats = d['features'].astype(np.float32)  # (N, n_feat)
            rul = d['rul'].astype(np.float32)
            self.segments.append((feats, rul, int(d['bearing_id'])))
            all_feats.append(feats)
        cat = np.concatenate(all_feats, axis=0)
        if normalize_stats is None:
            self.mean = cat.mean(0)
            self.std = cat.std(0) + 1e-6
        else:
            self.mean, self.std = normalize_stats
        self.windows = []  # (feats[w], rul_last, bearing_idx)
        for feats, rul, bid in self.segments:
            feats_norm = (feats - self.mean) / self.std
            for i in range(window - 1, len(feats)):
                self.windows.append((feats_norm[i-window+1:i+1].copy(), rul[i], bid))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        w, r, _ = self.windows[i]
        return torch.from_numpy(w), torch.tensor(r, dtype=torch.float32)

    def stats(self):
        return self.mean, self.std


class LSTM_RUL(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h[:, -1, :]).squeeze(-1)


def parse_oc(s):
    """'OC1,OC2' -> ['OC1','OC2']"""
    return [x.strip() for x in s.split(',') if x.strip()]


def collect_paths(cache_dir, ocs):
    """Return list of npz paths matching the given OCs (preprocessor uses OC1/OC2/OC3 in filename)."""
    out = []
    for oc in ocs:
        out.extend(sorted(glob.glob(os.path.join(cache_dir, f'xjtu_{oc}_*.npz'))))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache_dir', default=os.path.join(THIS_DIR, '..', 'data', 'xjtu_cache'))
    p.add_argument('--train_oc', default='OC1,OC2',
                   help='Operating conditions for training, comma-separated')
    p.add_argument('--test_oc',  default='OC3',
                   help='Operating condition for test')
    p.add_argument('--optimizer', default='LAKTJU_NS')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--window', type=int, default=10)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--num_layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--grad_clip', type=float, default=0.0)
    p.add_argument('--ns_interval', type=int, default=100)
    p.add_argument('--ns_steps', type=int, default=1)
    p.add_argument('--ns_max_dim', type=int, default=256)
    p.add_argument('--save_dir', default=os.path.join(THIS_DIR, '..', 'results_xjtu'))
    p.add_argument('--tag_suffix', default='')
    p.add_argument('--adaptive_trigger', action='store_true',
                   help='LAKTJU_NS only: trigger NS per-layer when estimated κ(M) > threshold.')
    p.add_argument('--kappa_threshold', type=float, default=1e4,
                   help='LAKTJU_NS adaptive trigger threshold (default 1e4).')
    p.add_argument('--kappa_gate', action='store_true',
                   help='LAKTJU_NS only: after the 100-step warm-up, screen per-kernel '
                        'kappa(M_t) and apply Newton-Schulz only to kernels with kappa>threshold.')
    args = p.parse_args()
    args.cache_dir = os.path.abspath(args.cache_dir)
    args.save_dir = os.path.abspath(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ocs = parse_oc(args.train_oc)
    test_ocs  = parse_oc(args.test_oc)
    train_paths = collect_paths(args.cache_dir, train_ocs)
    test_paths  = collect_paths(args.cache_dir, test_ocs)
    if not train_paths or not test_paths:
        print(f"ERROR: missing paths. train_paths={len(train_paths)} test_paths={len(test_paths)}", file=sys.stderr)
        sys.exit(2)

    # Hold out the last bearing of each train-OC as validation
    val_paths = []
    train_paths_filtered = []
    by_oc = {}
    for path in train_paths:
        oc_key = os.path.basename(path).split('_')[1]
        by_oc.setdefault(oc_key, []).append(path)
    for oc_key, paths in by_oc.items():
        paths.sort()
        val_paths.extend(paths[-1:])         # last bearing -> val
        train_paths_filtered.extend(paths[:-1])
    train_paths = train_paths_filtered

    print(f"[XJTU] train: {len(train_paths)} bearings ({train_ocs}); val: {len(val_paths)}; test: {len(test_paths)} ({test_ocs})")

    train_ds = XJTUDataset(train_paths, window=args.window)
    val_ds   = XJTUDataset(val_paths,   window=args.window, normalize_stats=train_ds.stats())
    test_ds  = XJTUDataset(test_paths,  window=args.window, normalize_stats=train_ds.stats())
    print(f"[XJTU] windows: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    n_feat = train_ds.windows[0][0].shape[-1]
    model = LSTM_RUL(input_dim=n_feat, hidden_dim=args.hidden_dim,
                     num_layers=args.num_layers, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LSTM_RUL: {n_params:,} parameters, input_dim={n_feat}")

    # Use build_optimizer from train_cmapss_rul (it uses args.lr/beta1/weight_decay/ns_*)
    optimizer, scheduler = _build_optimizer_from_cmapss(args, model)
    loss_fn = nn.MSELoss()

    # kappa-gate setup (no-op unless --kappa_gate and optimizer is LAKTJU_NS)
    named2d = [(nm, pp) for nm, pp in model.named_parameters() if pp.ndim >= 2]
    use_gate = getattr(args, 'kappa_gate', False) and args.optimizer == 'LAKTJU_NS'
    gate_done = not use_gate
    kappa_gate_log = None
    global_step = 0

    best_val_rmse = float('inf')
    best_test_rmse = float('inf')
    best_test_mae = float('inf')
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        tloss = 0.0; nb = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if (not gate_done) and global_step == args.ns_interval - 1:
                klog, skip = {}, []
                for nm, pp in named2d:
                    st = optimizer.state.get(pp, {})
                    kv = kappa_of(st['exp_avg']) if 'exp_avg' in st else None
                    klog[nm] = kv
                    if kv is not None and kv < args.kappa_threshold:
                        skip.append(pp)
                optimizer._ns_skip_ids = set(id(pp) for pp in skip)
                kappa_gate_log = {'measured_at_step': global_step + 1,
                                  'kappa_threshold': args.kappa_threshold,
                                  'per_kernel_kappa': klog,
                                  'skip_list': [nm for nm, pp in named2d
                                                if id(pp) in optimizer._ns_skip_ids],
                                  'n_kernels_2d': len(named2d), 'n_skipped': len(skip)}
                gate_done = True
                print(f"  [kappa-gate @ step {global_step+1}] skipped "
                      f"{len(skip)}/{len(named2d)}: {kappa_gate_log['skip_list']}")
            optimizer.step()
            global_step += 1
            tloss += float(loss) * yb.size(0); nb += yb.size(0)
        tloss /= max(nb, 1)

        def eval_loader(loader):
            model.eval()
            errs = []; abs_errs = []
            with torch.no_grad():
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    errs.append(((pred - yb) ** 2).cpu().numpy())
                    abs_errs.append((pred - yb).abs().cpu().numpy())
            sq = np.concatenate(errs); ae = np.concatenate(abs_errs)
            return float(np.sqrt(sq.mean())), float(ae.mean())

        val_rmse, val_mae = eval_loader(val_loader)
        test_rmse, test_mae = eval_loader(test_loader)
        if val_rmse < best_val_rmse:
            best_val_rmse, best_test_rmse, best_test_mae = val_rmse, test_rmse, test_mae

        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_rmse)
        else:
            try: scheduler.step()
            except Exception: pass

        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"[{epoch:3d}/{args.epochs}] train_loss={tloss:.3f} val_rmse={val_rmse:.3f} test_rmse={test_rmse:.3f} best={best_test_rmse:.3f}")
        history.append({'epoch': epoch, 'train_loss': tloss,
                        'val_rmse': val_rmse, 'val_mae': val_mae,
                        'test_rmse': test_rmse, 'test_mae': test_mae,
                        'best_val_rmse': best_val_rmse, 'best_test_rmse': best_test_rmse})

    final_test_rmse, final_test_mae = eval_loader(test_loader)
    out = {
        'config': vars(args),
        'final_test_rmse': final_test_rmse,
        'final_test_mae':  final_test_mae,
        'best_test_rmse':  best_test_rmse,
        'best_test_mae':   best_test_mae,
        'best_val_rmse':   best_val_rmse,
        'history':         history,
        'n_params':        n_params,
        'kappa_gate_log':  kappa_gate_log,
    }
    tag = args.tag_suffix or f"{args.optimizer}_seed{args.seed}"
    fname = f"xjtu_{'_'.join(train_ocs)}_to_{'_'.join(test_ocs)}_{tag}.json"
    out_path = os.path.join(args.save_dir, fname)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"Saved to {out_path}")
    print(f"Best test RMSE: {best_test_rmse:.3f}, MAE: {best_test_mae:.3f}")

if __name__ == '__main__':
    main()
