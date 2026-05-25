# FEMTO multi-partition robustness check (Plan B1)

Each (condition, optimizer) re-run at each partition's own best 36-config × 20 seeds. cond3 omitted (only 3 bearings → degenerate splits regardless of seed). Each partition draws a different held-out test bearing (5 partitions total).

| Cond | Opt | sp=2024 | sp=7 | sp=88 | sp=1 | sp=2 | replication |
|---|---|---|---|---|---|---|---|
| cond1 | PMO | **-2.12** (p=0.000, n=20) | **-14.63** (p=0.000, n=20) | +0.97 (p=0.292, n=20) | -0.24 (p=0.463, n=20) | +0.39 (p=0.133, n=20) | 2 sig win / 3 tie / 0 sig loss |
| cond1 | Adan | -0.51 (p=0.347, n=20) | **-7.74** (p=0.000, n=20) | +0.58 (p=0.756, n=20) | -0.29 (p=0.496, n=20) | **-0.92** (p=0.045, n=20) | 2 sig win / 3 tie / 0 sig loss |
| cond1 | RAdam | -0.44 (p=0.256, n=20) | **-9.77** (p=0.003, n=20) | -0.14 (p=0.885, n=20) | +1.94 (p=0.059, n=20) | **-0.97** (p=0.001, n=20) | 2 sig win / 3 tie / 0 sig loss |
| cond1 | Lion | **-1.43** (p=0.000, n=20) | **-13.74** (p=0.000, n=20) | -1.52 (p=0.291, n=20) | -0.02 (p=0.981, n=20) | **+2.21** (p=0.000, n=20) | 2 sig win / 2 tie / 1 sig loss |
| cond2 | PMO | **-13.92** (p=0.000, n=20) | **-22.26** (p=0.000, n=20) | **+1.09** (p=0.024, n=20) | +0.54 (p=0.462, n=20) | -0.25 (p=0.335, n=20) | 2 sig win / 2 tie / 1 sig loss |
| cond2 | Adan | **-5.70** (p=0.026, n=20) | **-7.55** (p=0.000, n=20) | **+2.04** (p=0.000, n=20) | +1.41 (p=0.077, n=20) | +0.14 (p=0.580, n=20) | 2 sig win / 2 tie / 1 sig loss |
| cond2 | RAdam | **-14.86** (p=0.000, n=20) | **-18.18** (p=0.000, n=20) | +0.38 (p=0.520, n=20) | -1.67 (p=0.278, n=20) | +0.70 (p=0.124, n=20) | 2 sig win / 3 tie / 0 sig loss |
| cond2 | Lion | -1.14 (p=0.579, n=20) | **-12.37** (p=0.044, n=20) | **+1.75** (p=0.000, n=20) | **+1.94** (p=0.007, n=20) | -0.01 (p=0.990, n=20) | 1 sig win / 2 tie / 2 sig loss |

## Headline (signed Δ in test RMSE; bold = significant p<0.05)

| Condition | Optimizer | wins replicate at | verdict |
|---|---|---|---|
| cond1 | PMO | sp[2024, 7] | PARTITION-CONDITIONAL WIN (2/5) |
| cond1 | Adan | sp[7, 2] | PARTITION-CONDITIONAL WIN (2/5) |
| cond1 | RAdam | sp[7, 2] | PARTITION-CONDITIONAL WIN (2/5) |
| cond1 | Lion | sp[2024, 7] | MIXED (2 win / 1 loss / 2 tie) |
| cond2 | PMO | sp[2024, 7] | MIXED (2 win / 1 loss / 2 tie) |
| cond2 | Adan | sp[2024, 7] | MIXED (2 win / 1 loss / 2 tie) |
| cond2 | RAdam | sp[2024, 7] | PARTITION-CONDITIONAL WIN (2/5) |
| cond2 | Lion | sp[7] | PARTIAL (1/5) |
