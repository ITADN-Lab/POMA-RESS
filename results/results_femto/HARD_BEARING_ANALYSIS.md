# FEMTO sp$=$88 hard-bearing characterization

Per-bearing summary statistics and per-partition train-pool distance.

## Per-bearing summary (cond1+cond2)

| Bearing | Life (acq) | Mean RMS | EoL burst ratio | RMS slope |
|---|---|---|---|---|
| Bearing1_1 | 2803 | 0.578 | 3.71 | 1.74e-03 |
| Bearing1_2 | 871 | 0.518 | 2.20 | 3.07e-03 |
| Bearing1_3 | 2375 | 0.839 | 7.43 | 3.22e-03 |
| Bearing1_4 | 1428 | 2.098 | 19.88 | 6.65e-03 |
| Bearing1_5 | 2463 | 0.377 | 1.19 | 5.10e-04 |
| Bearing1_6 | 2448 | 0.402 | 1.17 | 4.68e-04 |
| Bearing1_7 | 2259 | 0.453 | 1.85 | 8.69e-04 |
| Bearing2_1 | 911 | 0.683 | 1.64 | 2.09e-03 |
| Bearing2_2 | 797 | 0.719 | 1.84 | 2.63e-03 |
| Bearing2_3 | 1955 | 0.372 | 1.56 | 2.01e-03 |
| Bearing2_4 | 751 | 0.321 | 1.18 | 1.71e-03 |
| Bearing2_5 | 2311 | 0.373 | 0.84 | 3.56e-04 |
| Bearing2_6 | 701 | 0.311 | 1.63 | 2.18e-03 |
| Bearing2_7 | 230 | 0.449 | 3.15 | 1.44e-02 |

## Per-partition train/val/test assignment + held-out distance

| Cond | sp | train | val | test | dist(test, train centroid) |
|---|---|---|---|---|---|
| cond1 | 2024 | Bearing1_1,Bearing1_3,Bearing1_4,Bearing1_5,Bearing1_7 | Bearing1_6 | Bearing1_2 | 18.53 |
| cond1 | 7 | Bearing1_1,Bearing1_2,Bearing1_4,Bearing1_5,Bearing1_7 | Bearing1_6 | Bearing1_3 | 0.42 |
| cond1 | 88 | Bearing1_1,Bearing1_2,Bearing1_3,Bearing1_6,Bearing1_7 | Bearing1_5 | Bearing1_4 | 14.88 |
| cond2 | 2024 | Bearing2_1,Bearing2_3,Bearing2_4,Bearing2_5,Bearing2_7 | Bearing2_6 | Bearing2_2 | 3.69 |
| cond2 | 7 | Bearing2_1,Bearing2_2,Bearing2_4,Bearing2_5,Bearing2_7 | Bearing2_6 | Bearing2_3 | 11.63 |
| cond2 | 88 | Bearing2_1,Bearing2_2,Bearing2_3,Bearing2_6,Bearing2_7 | Bearing2_5 | Bearing2_4 | 2.85 |

## Rank of held-out test bearing's distance per condition

| Cond | sp=2024 distance | sp=7 distance | sp=88 distance | sp=88 rank? |
|---|---|---|---|---|
| cond1 | 18.53 | 0.42 | 14.88 | 2/3 |
| cond2 | 3.69 | 11.63 | 2.85 | 3/3 |
