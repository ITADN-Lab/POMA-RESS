#!/bin/bash
# 2.2 Non-LSTM architecture ablation on C-MAPSS FD001 & FD004
# Models: CNN1D, Transformer
# Optimizers: AdamW, LAKTJU_NS
# Seeds: 42, 123, 456

cd /home/hadoop/workstation/md/LafTJU-TII/experiments/scripts
SEEDS="42 123 456"

echo "=== 2.2 Architecture Ablation ==="
for subset in FD001 FD004; do
  for model in CNN1D Transformer; do
    for opt in AdamW LAKTJU_NS; do
      for seed in $SEEDS; do
        echo "[$subset/$model/$opt/seed$seed]"
        python3 train_cmapss_arch.py --subset $subset --model $model --optimizer $opt --seed $seed
      done
    done
  done
done
echo "=== 2.2 ALL DONE ==="
