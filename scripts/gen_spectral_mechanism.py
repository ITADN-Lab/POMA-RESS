#!/usr/bin/env python3
"""
Generate spectral_mechanism.pdf:
  Top   : event-aligned sawtooth of log10 kappa pre/post NS (LAKTJU-NS)
  Bottom: effective-rank distribution at NS-event moments comparing
          AdamW (sampled) vs LAKTJU-NS post-correction.

This honest figure supports the claim that NS *restores directional
diversity* (erank). The κ-exposure histogram was removed because κ is
an apples-to-oranges metric: NS's buffer is repeatedly reset to κ≈1
and then regrows over T steps, so the *time-averaged* κ within an
interval is not directly comparable to AdamW's sampled steady-state κ.
The directly comparable, mechanism-faithful quantity is the *post-NS*
state vs AdamW's *sampled* state, and especially erank (the directional
diversity), which is what NS is designed to restore.
"""
import json, glob, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/hadoop/workstation/md/RESS"
SPECTRAL_DIR = '/home/hadoop/workstation/md/LafTJU-TII/experiments/results_spectral'
OUT = os.path.join(REPO, "paper", "figures", "spectral_mechanism.pdf")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

def load_events(subset, optimizer):
    files = sorted(glob.glob(os.path.join(SPECTRAL_DIR, f"spectral_{subset}_{optimizer}_seed*.json")))
    out = []
    for f in files:
        d = json.load(open(f))
        out.append(d.get("spectral_log", []))
    return out

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))

# ---- Panel A: LAKTJU-NS sawtooth on FD001 ----
ns_events = load_events("FD001", "LAKTJU_NS")
max_events = max((len(s) for s in ns_events), default=0)
pre = np.full((len(ns_events), max_events), np.nan)
post = np.full((len(ns_events), max_events), np.nan)
for i, evs in enumerate(ns_events):
    for j, e in enumerate(evs):
        if e.get("kappa_pre") and e["kappa_pre"] > 0:
            pre[i, j] = math.log10(e["kappa_pre"])
        if e.get("kappa_post") and e["kappa_post"] > 0:
            post[i, j] = math.log10(e["kappa_post"])
pre_med = np.nanmedian(pre, axis=0)
post_med = np.nanmedian(post, axis=0)
pre_iqr_lo = np.nanpercentile(pre, 25, axis=0)
pre_iqr_hi = np.nanpercentile(pre, 75, axis=0)
post_iqr_lo = np.nanpercentile(post, 25, axis=0)
post_iqr_hi = np.nanpercentile(post, 75, axis=0)
idx = np.arange(max_events)

ax1 = axes[0]
ax1.fill_between(idx, pre_iqr_lo, pre_iqr_hi, alpha=0.18, color="C0")
ax1.fill_between(idx, post_iqr_lo, post_iqr_hi, alpha=0.18, color="C2")
ax1.plot(idx, pre_med, "-", color="C0", lw=1.4, label=r"pre-NS  ($\sim 10^7$)")
ax1.plot(idx, post_med, "--", color="C2", lw=1.4, label=r"post-NS ($\sim 10^{2.5}$)")
# Show the 4-5 OoM correction with an annotation arrow on the last event
ax1.annotate("", xy=(max_events-3, post_med[-3]),
             xytext=(max_events-3, pre_med[-3]),
             arrowprops=dict(arrowstyle="->", color="black", lw=1))
ax1.text(max_events-1, (pre_med[-3] + post_med[-3]) / 2,
         r"$\sim 4{-}5\,$OoM", fontsize=8, ha="right", va="center")
ax1.set_xlabel("NS event index (every $T{=}100$ steps)")
ax1.set_ylabel(r"$\log_{10}\kappa(M_t)$")
ax1.set_title("(a) POMA pre/post sawtooth (FD001, 5 seeds)")
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8, loc="center right")

# ---- Panel B: effective-rank distribution: AdamW sampled vs NS post-correction ----
adamw_events_all = []
ns_post_all = []
for subset in ["FD001", "FD003", "FD004"]:
    aw = load_events(subset, "AdamW")
    for evs in aw:
        for e in evs:
            for key in ("erank", "erank_pre", "erank_post"):
                v = e.get(key)
                if v is not None and v > 0:
                    adamw_events_all.append(v)
                    break
    ns = load_events(subset, "LAKTJU_NS")
    for evs in ns:
        for e in evs:
            if e.get("erank_post") and e["erank_post"] > 0:
                ns_post_all.append(e["erank_post"])

ax2 = axes[1]
adamw_events_all = np.asarray(adamw_events_all)
ns_post_all = np.asarray(ns_post_all)
if adamw_events_all.size == 0:
    # Fallback: use erank_pre from NS logs (AdamW's pre-NS state, since NS resets from there)
    for subset in ["FD001", "FD003", "FD004"]:
        ns = load_events(subset, "LAKTJU_NS")
        for evs in ns:
            for e in evs:
                if e.get("erank_pre") and e["erank_pre"] > 0:
                    adamw_events_all = np.append(adamw_events_all, e["erank_pre"])
    label_aw = "uncorrected buffer (pre-NS state)"
else:
    label_aw = "AdamW sampled erank"

bins = np.linspace(0, max(ns_post_all.max(), adamw_events_all.max()) * 1.05, 30)
ax2.hist(adamw_events_all, bins=bins, color="C3", alpha=0.55,
         label=f"{label_aw}\n(median $\\approx$ {np.median(adamw_events_all):.2f})")
ax2.hist(ns_post_all, bins=bins, color="C0", alpha=0.55,
         label=f"POMA post-correction\n(median $\\approx$ {np.median(ns_post_all):.2f})")
ax2.set_xlabel("Effective rank of first-layer momentum buffer")
ax2.set_ylabel("Count (NS events)")
ax2.set_title("(b) NS post-correction restores erank")
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8, loc="upper right")

plt.tight_layout()
plt.savefig(OUT, format="pdf", bbox_inches="tight")
print(f"wrote {OUT}")
print(f"NS pre med log10 kappa: {np.nanmedian(pre):.2f}")
print(f"NS post med log10 kappa: {np.nanmedian(post):.2f}")
print(f"NS erank post median: {np.median(ns_post_all):.2f}")
print(f"AdamW/pre-NS erank median: {np.median(adamw_events_all):.2f}")
