#!/bin/bash
cd /home/hadoop/workstation/md/LafTJU-TII/experiments/scripts
SEEDS="42 123 456"
OPTS="AdamW LAKTJU_NS Shampoo SOAP"
echo "=== 2.4 Second-Order Comparison on FD001 ==="
for opt in $OPTS; do
  for seed in $SEEDS; do
    echo "[$opt seed=$seed]"
    python3 run_ablation_2_4.py --subset FD001 --optimizer $opt --seed $seed 2>&1 | tail -1
  done
done
echo "=== 2.4 ALL DONE ==="
