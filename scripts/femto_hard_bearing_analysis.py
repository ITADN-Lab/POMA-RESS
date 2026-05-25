"""
FEMTO sp=88 hard-bearing characterization (Plan B Codex priority).

For each leave-bearings-out partition we want to explain *why* sp=88 is
"hard": which bearing(s) does it hold out, what is the held-out bearing's
distance from the training pool, and what physical / statistical feature
makes it an outlier.

Reads FEMTO bearing cache (npy) and prints a per-bearing summary table +
a per-partition (train pool centroid, held-out test bearing) distance.
"""
import os, glob, json
import numpy as np

CACHE_DIR = os.path.expanduser('~/pmo_femto/data/src')
RESULTS = os.path.expanduser('~/pmo_femto/results')
OUT_MD = os.path.expanduser('~/pmo_femto/results/HARD_BEARING_ANALYSIS.md')


def load_bearing_mat(b):
    """Look in standard subdirs; reuse the cache file the trainer wrote."""
    for sub in ('Learning_set', 'Full_Test_Set', 'Test_set'):
        p = os.path.join(CACHE_DIR, sub, b)
        if os.path.isdir(p):
            cache = os.path.join(CACHE_DIR, f'_cache_{b}.npy')
            if os.path.isfile(cache):
                return np.load(cache)
    return None


def bearing_summary(b, mat):
    """Return scalar features summarising the bearing's degradation."""
    if mat is None or len(mat) < 50:
        return None
    n = len(mat)
    # cols: 0=rms_h, 1=rms_v, 2=kurt_h, 3=kurt_v, 4=p2p_h, 5=p2p_v
    rms = np.mean(mat[:, [0, 1]], axis=1)
    p2p = np.mean(mat[:, [4, 5]], axis=1)
    # life is the acquisition count (each ~10 s)
    # end-of-life burst: ratio of peak vs steady-state
    steady = rms[: int(0.5 * n)].mean()
    burst = rms[int(0.9 * n) :].mean()
    slope = (rms[-1] - rms[0]) / max(1, n)
    return {
        'life_n': n,
        'mean_rms': float(rms.mean()),
        'eol_burst_ratio': float(burst / max(steady, 1e-6)),
        'rms_slope_per_acq': float(slope),
        'mean_p2p': float(p2p.mean()),
        'feature_vec': mat.mean(axis=0).tolist(),
    }


def partition_split(n_bearings, sp):
    """Reproduce build_femto_datasets train/val/test partition."""
    perm = np.random.RandomState(sp).permutation(n_bearings)
    n_val = max(1, int(round(0.20 * n_bearings)))
    n_test = max(1, int(round(0.20 * n_bearings)))
    test_idx = sorted(perm[:n_test].tolist())
    val_idx = sorted(perm[n_test:n_test + n_val].tolist())
    train_idx = sorted(perm[n_test + n_val:].tolist())
    return train_idx, val_idx, test_idx


def main():
    CONDS = {
        'cond1': ['Bearing1_1', 'Bearing1_2', 'Bearing1_3', 'Bearing1_4',
                  'Bearing1_5', 'Bearing1_6', 'Bearing1_7'],
        'cond2': ['Bearing2_1', 'Bearing2_2', 'Bearing2_3', 'Bearing2_4',
                  'Bearing2_5', 'Bearing2_6', 'Bearing2_7'],
    }
    SPS = [2024, 7, 88]

    md = ["# FEMTO sp$=$88 hard-bearing characterization\n",
          "Per-bearing summary statistics and per-partition train-pool distance.\n",
          "## Per-bearing summary (cond1+cond2)\n",
          "| Bearing | Life (acq) | Mean RMS | EoL burst ratio | RMS slope |",
          "|---|---|---|---|---|"]
    summaries = {}
    for cond, bs in CONDS.items():
        for b in bs:
            mat = load_bearing_mat(b)
            s = bearing_summary(b, mat)
            summaries[b] = s
            if s:
                md.append(f"| {b} | {s['life_n']} | {s['mean_rms']:.3f} | "
                          f"{s['eol_burst_ratio']:.2f} | {s['rms_slope_per_acq']:.2e} |")
    md.append("")

    md.append("## Per-partition train/val/test assignment + held-out distance\n")
    md.append("| Cond | sp | train | val | test | dist(test, train centroid) |")
    md.append("|---|---|---|---|---|---|")
    sp_dist = {}
    for cond, bs in CONDS.items():
        for sp in SPS:
            ti, vi, te = partition_split(len(bs), sp)
            train_bs = [bs[i] for i in ti]
            val_bs = [bs[i] for i in vi]
            test_bs = [bs[i] for i in te]
            train_feats = np.array([summaries[b]['feature_vec'] for b in train_bs
                                    if summaries.get(b)])
            test_feats = np.array([summaries[b]['feature_vec'] for b in test_bs
                                   if summaries.get(b)])
            if len(train_feats) == 0 or len(test_feats) == 0:
                md.append(f"| {cond} | {sp} | {train_bs} | {val_bs} | {test_bs} | — |")
                continue
            tr_centroid = train_feats.mean(axis=0)
            tr_std = train_feats.std(axis=0) + 1e-9
            z = (test_feats[0] - tr_centroid) / tr_std
            d = float(np.sqrt(float(np.sum(z * z))))
            sp_dist[(cond, sp)] = d
            md.append(f"| {cond} | {sp} | {','.join(train_bs)} | {','.join(val_bs)} | {','.join(test_bs)} | {d:.2f} |")

    md.append("")
    md.append("## Rank of held-out test bearing's distance per condition\n")
    md.append("| Cond | sp=2024 distance | sp=7 distance | sp=88 distance | sp=88 rank? |")
    md.append("|---|---|---|---|---|")
    for cond in CONDS:
        ds = {sp: sp_dist.get((cond, sp), float('nan')) for sp in SPS}
        ranked = sorted([(sp, d) for sp, d in ds.items()], key=lambda x: -x[1])
        sp88_rank = next((i+1 for i, (sp, _) in enumerate(ranked) if sp == 88), '—')
        md.append(f"| {cond} | {ds[2024]:.2f} | {ds[7]:.2f} | {ds[88]:.2f} | "
                  f"{sp88_rank}/{len(SPS)} |")

    text = "\n".join(md) + "\n"
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    open(OUT_MD, 'w').write(text)
    print(text)
    print(f"wrote {OUT_MD}")


if __name__ == '__main__':
    main()
