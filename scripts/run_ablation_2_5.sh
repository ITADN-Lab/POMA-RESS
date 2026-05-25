#!/bin/bash
# 2.5 Long training stability: 500 epochs on C-MAPSS FD001
cd /home/hadoop/workstation/md/LafTJU-TII/experiments/scripts
SEEDS="42 123 456"
SAVE_DIR="../results/ablation_2_5"
mkdir -p $SAVE_DIR

echo "=== 2.5 Long Training (500 epochs) on FD001 ==="
for opt in AdamW LAKTJU_NS; do
  for seed in $SEEDS; do
    echo "[$opt seed=$seed]"
    python3 train_cmapss_rul.py \
      --subset FD001 --optimizer $opt --seed $seed \
      --epochs 500 --grad_clip 0.0 --ns_interval 100 \
      --save_dir $SAVE_DIR 2>&1 | tail -1
  done
done
echo "=== 2.5 ALL DONE ==="
