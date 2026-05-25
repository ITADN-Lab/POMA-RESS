# NASA Battery audit panel (Plan B v2 minimal replication)

Cross-domain (electrochemical, vs C-MAPSS turbofan + FEMTO bearings) — 4 candidates × 2 partitions × 3 seeds. Per-partition Phase A (12-config grid) picks each opt's best, then 3 seeds × best-config paired vs AdamW.

| Opt | sp | AdamW (mean) | Opt (mean) | Δ | 95% CI | p | dz | n |
|---|---|---|---|---|---|---|---|---|
| PMO | 2024 | 30.82 | 35.21 | **+4.39** | [+0.44, +10.54] | 0.000 | +0.81 | 3 |
| Adan | 2024 | 30.82 | 26.92 | -3.90 | [-10.72, +9.32] | 0.521 | -0.34 | 3 |
| RAdam | 2024 | 30.82 | 43.80 | **+12.98** | [+8.36, +18.84] | 0.000 | +2.43 | 3 |
| Lion | 2024 | 30.82 | 43.04 | **+12.22** | [+9.31, +16.54] | 0.000 | +3.20 | 3 |
| PMO | 7 | 27.80 | 30.74 | +2.94 | [-5.38, +15.76] | 0.586 | +0.26 | 3 |
| Adan | 7 | 27.80 | 22.36 | -5.44 | [-12.51, +8.25] | 0.521 | -0.46 | 3 |
| RAdam | 7 | 27.80 | 36.65 | **+8.85** | [+2.49, +18.03] | 0.000 | +1.09 | 3 |
| Lion | 7 | 27.80 | 43.99 | **+16.19** | [+9.37, +25.95] | 0.000 | +1.87 | 3 |

## Headline (Δ vs AdamW, bold = sig p<0.05)

| Opt | sp=2024 | sp=7 |
|---|---|---|
| PMO | **+4.39** (p=0.000) | +2.94 (p=0.586) |
| Adan | -3.90 (p=0.521) | -5.44 (p=0.521) |
| RAdam | **+12.98** (p=0.000) | **+8.85** (p=0.000) |
| Lion | **+12.22** (p=0.000) | **+16.19** (p=0.000) |
