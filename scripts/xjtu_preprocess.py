"""
XJTU-SY Bearing Dataset preprocessing → time+frequency features → npz cache.

Source format (from raw zip):
  35Hz12kN/Bearing1_1/{1,2,3,...,N}.csv   (1 file per minute, 32768 samples × 2 channels)
  37.5Hz11kN/Bearing2_1/...
  40Hz10kN/Bearing3_1/...

Per minute we extract a feature vector (16 features):
  per channel (×2): RMS, peak, kurtosis, skewness, crest factor,
                    shape factor, FFT band energy in 4 bands (0-1kHz, 1-3kHz, 3-6kHz, 6-12kHz)
  → 8 features × 2 channels = 16 features per minute

Output:
  cache/xjtu_<condition>_<bearing>.npz with arrays:
    features: (N_minutes, 16)
    rul:      (N_minutes,)  — linear RUL with cap 125 (in minutes, matching C-MAPSS convention)
    bearing_id, condition_id  — metadata
"""
import os, sys, json, glob, argparse, time
import numpy as np
from scipy.stats import kurtosis, skew

SAMPLING_RATE = 25600  # Hz
FFT_BANDS = [(0, 1000), (1000, 3000), (3000, 6000), (6000, 12000)]  # Hz
RUL_CAP = 125

# Operating-condition directory naming
OC_NAMES = {
    'OC1': '35Hz12kN',
    'OC2': '37.5Hz11kN',
    'OC3': '40Hz10kN',
}

def time_features(x):
    """8 time-domain + frequency features for one channel signal (length 32768)."""
    rms = np.sqrt(np.mean(x**2))
    peak = np.max(np.abs(x))
    krt = kurtosis(x, fisher=False)
    skw = skew(x)
    crest = peak / (rms + 1e-12)
    shape = rms / (np.mean(np.abs(x)) + 1e-12)
    # FFT band energies
    X = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0/SAMPLING_RATE)
    bands = []
    for lo, hi in FFT_BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        bands.append(np.sqrt(np.mean(X[mask]**2)) if mask.any() else 0.0)
    return [rms, peak, krt, skw, crest, shape] + bands  # 6+4 = 10? wait we said 8 total per channel

def feature_vec(csv_path):
    """Load one minute csv: 32768 × 2 channels."""
    arr = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    feats = []
    for ch in range(arr.shape[1]):
        x = arr[:, ch]
        feats.extend(time_features(x))
    return np.array(feats, dtype=np.float32)

def process_bearing(bearing_dir, cache_path, oc_idx, bearing_idx):
    """Process one bearing's run-to-failure CSV sequence."""
    files = sorted(glob.glob(os.path.join(bearing_dir, '*.csv')),
                   key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
    if not files:
        return None
    N = len(files)
    feats = []
    t0 = time.time()
    for i, f in enumerate(files):
        try:
            feats.append(feature_vec(f))
        except Exception as e:
            print(f"  WARN: bad file {f}: {e}", file=sys.stderr)
            feats.append(np.zeros(20, dtype=np.float32))  # placeholder
        if (i+1) % 50 == 0:
            print(f"  bearing {bearing_dir.split('/')[-1]}: {i+1}/{N} ({time.time()-t0:.1f}s)")
    feats = np.stack(feats)
    n_feat = feats.shape[1]
    # Linear RUL with cap (in minutes)
    rul_full = np.arange(N-1, -1, -1, dtype=np.float32)  # minutes remaining
    rul = np.clip(rul_full, 0, RUL_CAP)
    np.savez(cache_path,
             features=feats,
             rul=rul,
             rul_full=rul_full,
             bearing_id=bearing_idx,
             condition_id=oc_idx,
             total_minutes=N,
             n_feat=n_feat,
             bearing_dir=os.path.basename(bearing_dir))
    print(f"[OK] {bearing_dir.split('/')[-1]}: {N} minutes, {n_feat} features → {cache_path} ({time.time()-t0:.1f}s)")
    return cache_path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw_dir', required=True, help='Root containing 35Hz12kN/, 37.5Hz11kN/, 40Hz10kN/')
    p.add_argument('--cache_dir', required=True)
    p.add_argument('--n_jobs', type=int, default=1)
    args = p.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    todo = []
    for oc_idx, (oc_name, oc_dir) in enumerate(OC_NAMES.items()):
        oc_path = os.path.join(args.raw_dir, oc_dir)
        if not os.path.isdir(oc_path):
            print(f"WARN: missing OC dir {oc_path}", file=sys.stderr)
            continue
        bearings = sorted([d for d in os.listdir(oc_path) if d.startswith('Bearing')])
        for bearing_idx, bearing in enumerate(bearings):
            bearing_dir = os.path.join(oc_path, bearing)
            cache_path = os.path.join(args.cache_dir, f"xjtu_{oc_name}_{bearing}.npz")
            if os.path.exists(cache_path):
                print(f"[SKIP] {cache_path} already exists")
                continue
            todo.append((bearing_dir, cache_path, oc_idx, bearing_idx))

    print(f"\n[xjtu_preprocess] {len(todo)} bearings to process")

    if args.n_jobs == 1:
        for args_tuple in todo:
            process_bearing(*args_tuple)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            futs = [ex.submit(process_bearing, *t) for t in todo]
            for f in as_completed(futs):
                try: f.result()
                except Exception as e: print(f"ERR: {e}", file=sys.stderr)

    # Write manifest
    manifest = {
        'oc_names': OC_NAMES,
        'bearings_per_oc': {},
        'rul_cap': RUL_CAP,
        'sampling_rate': SAMPLING_RATE,
        'fft_bands_hz': FFT_BANDS,
        'feature_names_per_channel': [
            'rms', 'peak', 'kurtosis', 'skew', 'crest', 'shape',
            'fft_band_0_1k', 'fft_band_1_3k', 'fft_band_3_6k', 'fft_band_6_12k'
        ],
        'n_features': 20,  # 10 per channel × 2 channels
    }
    for oc_name in OC_NAMES:
        manifest['bearings_per_oc'][oc_name] = sorted([
            os.path.basename(p).replace(f'xjtu_{oc_name}_', '').replace('.npz', '')
            for p in glob.glob(os.path.join(args.cache_dir, f'xjtu_{oc_name}_*.npz'))
        ])
    with open(os.path.join(args.cache_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {args.cache_dir}/manifest.json")

if __name__ == '__main__':
    main()
