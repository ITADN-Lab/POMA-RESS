# FEMTO Plan-B variance decomposition + sp=88 val/test gap

## Per-partition val/test RMSE gap (distribution-shift proxy)

| Cond | Opt | sp | val mean | test mean | val/test ratio |
|---|---|---|---|---|---|
| cond1 | PMO | 2024 | 15.56 | 26.29 | 1.69× |
| cond1 | PMO | 7 | 14.33 | 21.87 | 1.53× |
| cond1 | PMO | 88 | 5.89 | 43.12 | 7.32× |
| cond1 | PMO | 1 | 6.74 | 15.66 | 2.32× |
| cond1 | PMO | 2 | 16.72 | 13.51 | 0.81× |
| cond1 | Adan | 2024 | 15.25 | 27.90 | 1.83× |
| cond1 | Adan | 7 | 11.07 | 28.77 | 2.60× |
| cond1 | Adan | 88 | 6.69 | 42.74 | 6.39× |
| cond1 | Adan | 1 | 11.54 | 15.61 | 1.35× |
| cond1 | Adan | 2 | 15.57 | 12.20 | 0.78× |
| cond1 | RAdam | 2024 | 15.05 | 27.97 | 1.86× |
| cond1 | RAdam | 7 | 11.50 | 26.74 | 2.32× |
| cond1 | RAdam | 88 | 5.11 | 42.01 | 8.22× |
| cond1 | RAdam | 1 | 6.67 | 17.83 | 2.67× |
| cond1 | RAdam | 2 | 17.11 | 12.15 | 0.71× |
| cond1 | Lion | 2024 | 16.14 | 26.98 | 1.67× |
| cond1 | Lion | 7 | 14.96 | 22.77 | 1.52× |
| cond1 | Lion | 88 | 9.33 | 40.64 | 4.35× |
| cond1 | Lion | 1 | 14.07 | 15.88 | 1.13× |
| cond1 | Lion | 2 | 22.08 | 15.33 | 0.69× |
| cond1 | AdamW | 2024 | 14.97 | 28.41 | 1.90× |
| cond1 | AdamW | 7 | 10.77 | 36.50 | 3.39× |
| cond1 | AdamW | 88 | 4.81 | 42.15 | 8.76× |
| cond1 | AdamW | 1 | 9.90 | 15.90 | 1.61× |
| cond1 | AdamW | 2 | 16.31 | 13.12 | 0.80× |
| cond2 | PMO | 2024 | 23.75 | 27.91 | 1.18× |
| cond2 | PMO | 7 | 27.03 | 49.18 | 1.82× |
| cond2 | PMO | 88 | 16.59 | 26.87 | 1.62× |
| cond2 | PMO | 1 | 20.39 | 52.39 | 2.57× |
| cond2 | PMO | 2 | 24.86 | 17.67 | 0.71× |
| cond2 | Adan | 2024 | 24.20 | 36.13 | 1.49× |
| cond2 | Adan | 7 | 25.63 | 63.89 | 2.49× |
| cond2 | Adan | 88 | 16.70 | 27.82 | 1.67× |
| cond2 | Adan | 1 | 19.12 | 53.26 | 2.78× |
| cond2 | Adan | 2 | 25.41 | 18.05 | 0.71× |
| cond2 | RAdam | 2024 | 24.19 | 26.96 | 1.11× |
| cond2 | RAdam | 7 | 27.74 | 53.26 | 1.92× |
| cond2 | RAdam | 88 | 16.15 | 26.16 | 1.62× |
| cond2 | RAdam | 1 | 29.74 | 50.18 | 1.69× |
| cond2 | RAdam | 2 | 22.83 | 18.62 | 0.82× |
| cond2 | Lion | 2024 | 23.98 | 40.68 | 1.70× |
| cond2 | Lion | 7 | 26.01 | 59.07 | 2.27× |
| cond2 | Lion | 88 | 16.41 | 27.53 | 1.68× |
| cond2 | Lion | 1 | 18.98 | 53.79 | 2.83× |
| cond2 | Lion | 2 | 25.74 | 17.91 | 0.70× |
| cond2 | AdamW | 2024 | 23.69 | 41.82 | 1.77× |
| cond2 | AdamW | 7 | 25.66 | 71.44 | 2.78× |
| cond2 | AdamW | 88 | 16.46 | 25.78 | 1.57× |
| cond2 | AdamW | 1 | 22.23 | 51.85 | 2.33× |
| cond2 | AdamW | 2 | 24.17 | 17.92 | 0.74× |

## Variance decomposition (per cond, opt): how much of $\Delta$ uncertainty is partition vs seed

| Cond | Opt | sd_seed (mean over partitions) | sd_partition (across mean-Δ) | partition/seed ratio |
|---|---|---|---|---|
| cond1 | PMO | 4.51 | 6.54 | 1.45× |
| cond1 | Adan | 5.18 | 3.38 | 0.65× |
| cond1 | RAdam | 5.42 | 4.55 | 0.84× |
| cond1 | Lion | 5.26 | 6.24 | 1.19× |
| cond2 | PMO | 7.61 | 10.59 | 1.39× |
| cond2 | Adan | 5.15 | 4.39 | 0.85× |
| cond2 | RAdam | 8.56 | 9.06 | 1.06× |
| cond2 | Lion | 8.78 | 5.95 | 0.68× |


*Partition variance > seed variance means optimizer apparent advantage is dominated by which bearing is held out.*

