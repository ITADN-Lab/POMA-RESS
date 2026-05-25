"""
Generate the naive-Δ → leak-free-Δ arrow visualisation for the POMA case study.
For each C-MAPSS subset, draws an arrow from the naive ∆(POMA−AdamW) to the
leak-free 20-seed Δ(POMA−AdamW). Significant Δs are bold-coloured.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(
                   '/home/hadoop/workstation/md/RESS/paper/figures', 'audit_arrow.pdf')

# (subset, naive Δ, naive p, leak-free Δ, leak-free p)
DATA = [
    ('FD001', +0.05, 0.82,  +0.02, 0.886),
    ('FD002', -1.41, 0.001, +0.20, 0.014),
    ('FD003', +0.43, 0.16,  +0.53, 0.137),
    ('FD004', -1.24, 0.006, -0.68, 0.002),
]


def main():
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    y_positions = np.arange(len(DATA))[::-1]
    for y, (name, dn, pn, df, pf) in zip(y_positions, DATA):
        # Naive endpoint
        col_n = '#c0392b' if pn < 0.05 else '#999999'
        ax.scatter([dn], [y], s=70, color=col_n, zorder=3,
                   edgecolor='black', linewidth=0.6)
        ax.annotate(f"{dn:+.2f}", (dn, y), xytext=(0, 7),
                    textcoords='offset points', ha='center',
                    fontsize=8, color=col_n,
                    fontweight='bold' if pn < 0.05 else 'normal')
        # Leak-free endpoint
        col_f = '#2980b9' if pf < 0.05 else '#999999'
        ax.scatter([df], [y], s=70, color=col_f, zorder=3,
                   edgecolor='black', linewidth=0.6)
        ax.annotate(f"{df:+.2f}", (df, y), xytext=(0, 7),
                    textcoords='offset points', ha='center',
                    fontsize=8, color=col_f,
                    fontweight='bold' if pf < 0.05 else 'normal')
        # Arrow
        ax.annotate('', xy=(df, y), xytext=(dn, y),
                    arrowprops=dict(arrowstyle='->', lw=1.1,
                                    color='#555555', alpha=0.85))
        ax.text(-2.0, y, name, ha='right', va='center', fontsize=9)

    ax.axvline(0, color='black', lw=0.6, alpha=0.6)
    ax.set_xlim(-2.0, 1.2)
    ax.set_ylim(-0.6, len(DATA) - 0.4)
    ax.set_xlabel(r'$\Delta$(POMA $-$ AdamW)  test RMSE  (lower = POMA better)',
                  fontsize=9)
    ax.set_yticks([])
    ax.grid(axis='x', alpha=0.3)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)

    # Legend
    ax.scatter([], [], s=70, color='#c0392b', edgecolor='black',
               linewidth=0.6, label='Naive protocol (significant)')
    ax.scatter([], [], s=70, color='#2980b9', edgecolor='black',
               linewidth=0.6, label='Leak-free 20-seed (significant)')
    ax.scatter([], [], s=70, color='#999999', edgecolor='black',
               linewidth=0.6, label='Not significant')
    ax.legend(loc='upper right', fontsize=7, frameon=False,
              handletextpad=0.4, columnspacing=0.8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches='tight')
    print(f"wrote {OUT}")


if __name__ == '__main__':
    main()
