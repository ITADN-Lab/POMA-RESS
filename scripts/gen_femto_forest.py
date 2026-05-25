"""
Generate a forest/heatmap figure showing FEMTO per-(cond, opt, partition) Δ
to make the sp88 hard-partition heterogeneity first-class evidence.

Reads experiments/results/results_femto/femto_multipart_summary.json and produces
paper/figures/femto_partition_heterogeneity.pdf with two panels:

  Left  : forest plot of Δ ± 95% CI for each (cond, opt, partition) cell
  Right : heatmap of Δ (signed) over conditions × optimizers × partitions
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THIS = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(THIS, '..', 'results/results_femto')
FIG_OUT = os.path.join(THIS, '..', '..', 'paper', 'figures',
                       'femto_partition_heterogeneity.pdf')

PARTITIONS = ['2024', '7', '88', '1', '2']
OPTS = ['POMA', 'Adan', 'RAdam', 'Lion']
CONDS = ['cond1', 'cond2']

data = json.load(open(os.path.join(RES, 'femto_multipart_summary.json')))

fig, ax = plt.subplots(1, 2, figsize=(8.8, 5.0),
                       gridspec_kw={'width_ratios': [1.55, 1.0]})

# ----- Left: forest plot -----
rows = []   # (label, delta, lo, hi, color)
for cond in CONDS:
    for opt in OPTS:
        for sp in PARTITIONS:
            st = data.get(cond, {}).get(opt, {}).get(sp)
            if not st: continue
            ci = st['ci']
            d = st['delta']
            p = st['p_boot']
            label = f"{cond}/{opt}/sp{sp}"
            color = ('tab:green' if d < 0 and p < 0.05
                     else ('tab:red' if d > 0 and p < 0.05
                           else 'tab:gray'))
            rows.append((label, d, ci[0], ci[1], color, p))

# put cond1 on top, cond2 below; group by opt
y = np.arange(len(rows))[::-1]
for i, (label, d, lo, hi, c, p) in enumerate(rows):
    ax[0].errorbar(d, y[i], xerr=[[d - lo], [hi - d]],
                   fmt='o', color=c, capsize=2.5, markersize=4)
ax[0].axvline(0, color='k', lw=0.6, linestyle='--')
ax[0].set_yticks(y)
ax[0].set_yticklabels([r[0] for r in rows], fontsize=7)
ax[0].set_xlabel(r'$\Delta$(optimizer $-$ AdamW), test-RMSE')
ax[0].set_title('FEMTO multi-partition heterogeneity\n(green=sig win, red=sig loss, grey=tie)',
                fontsize=9)
ax[0].grid(axis='x', alpha=0.3)

# ----- Right: heatmap -----
mat = np.full((len(CONDS) * len(OPTS), len(PARTITIONS)), np.nan)
for ci, cond in enumerate(CONDS):
    for oi, opt in enumerate(OPTS):
        for pi, sp in enumerate(PARTITIONS):
            st = data.get(cond, {}).get(opt, {}).get(sp)
            if st:
                mat[ci * len(OPTS) + oi, pi] = st['delta']
vmax = np.nanmax(np.abs(mat))
im = ax[1].imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
ax[1].set_xticks(range(len(PARTITIONS)))
ax[1].set_xticklabels([f'sp={p}\n{"(hard)" if p=="88" else ""}' for p in PARTITIONS],
                      fontsize=8)
labels = []
for cond in CONDS:
    for opt in OPTS:
        labels.append(f'{cond}/{opt}')
ax[1].set_yticks(range(len(labels)))
ax[1].set_yticklabels(labels, fontsize=8)
ax[1].set_title(r'$\Delta$(opt$-$AdamW) heatmap', fontsize=9)
# annotate cells with Δ values
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        if not np.isnan(mat[i, j]):
            txt = f'{mat[i, j]:+.1f}'
            color = 'white' if abs(mat[i, j]) > 0.5 * vmax else 'black'
            ax[1].text(j, i, txt, ha='center', va='center',
                       fontsize=7, color=color)
fig.colorbar(im, ax=ax[1], shrink=0.85, label=r'$\Delta$ RMSE')

# vertical separators between conditions on the heatmap
ax[1].axhline(len(OPTS) - 0.5, color='k', lw=0.7)

plt.tight_layout()
os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
plt.savefig(FIG_OUT, bbox_inches='tight')
print(f"wrote {FIG_OUT}")
plt.close()
