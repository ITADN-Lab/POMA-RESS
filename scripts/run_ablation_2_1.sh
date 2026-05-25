#!/bin/bash
# 2.1 Ablation experiments on C-MAPSS FD001
# Run on 12号 machine
PYTHON=/home/hadoop/laftju_env/bin/python
cd /home/hadoop/LafTJU-TII/experiments/scripts
SEEDS="42 123 456"
SAVE_DIR="../results/ablation_2_1"
mkdir -p $SAVE_DIR

echo "=== (a) NS Interval Sweep ==="
for interval in 10 25 50 100 200 500 1000; do
  for seed in $SEEDS; do
    echo "[interval=$interval seed=$seed]"
    $PYTHON train_cmapss_rul.py \
      --subset FD001 --optimizer LAKTJU_NS --seed $seed \
      --ns_interval $interval --ns_steps 1 \
      --save_dir $SAVE_DIR --tag_suffix "int${interval}" 2>&1 | tail -1
  done
done

echo "=== (b) GC x NS 2x2 ==="
echo "  GC=on, NS=on"
for seed in $SEEDS; do
  echo "[gc1_ns1 seed=$seed]"
  $PYTHON train_cmapss_rul.py \
    --subset FD001 --optimizer LAKTJU_NS --seed $seed \
    --ns_interval 100 --grad_clip 1.0 \
    --save_dir $SAVE_DIR --tag_suffix "gc1_ns1" 2>&1 | tail -1
done

echo "  GC=off, NS=on"
for seed in $SEEDS; do
  echo "[gc0_ns1 seed=$seed]"
  $PYTHON train_cmapss_rul.py \
    --subset FD001 --optimizer LAKTJU_NS --seed $seed \
    --ns_interval 100 --grad_clip 0.0 \
    --save_dir $SAVE_DIR --tag_suffix "gc0_ns1" 2>&1 | tail -1
done

echo "  GC=on, NS=off"
for seed in $SEEDS; do
  echo "[gc1_ns0 seed=$seed]"
  $PYTHON train_cmapss_rul.py \
    --subset FD001 --optimizer AdamW --seed $seed \
    --grad_clip 1.0 \
    --save_dir $SAVE_DIR --tag_suffix "gc1_ns0" 2>&1 | tail -1
done

echo "  GC=off, NS=off"
for seed in $SEEDS; do
  echo "[gc0_ns0 seed=$seed]"
  $PYTHON train_cmapss_rul.py \
    --subset FD001 --optimizer AdamW --seed $seed \
    --grad_clip 0.0 \
    --save_dir $SAVE_DIR --tag_suffix "gc0_ns0" 2>&1 | tail -1
done

echo "=== (c) NS Steps Sweep ==="
for steps in 1 2 3; do
  for seed in $SEEDS; do
    echo "[ns_steps=$steps seed=$seed]"
    $PYTHON train_cmapss_rul.py \
      --subset FD001 --optimizer LAKTJU_NS --seed $seed \
      --ns_interval 100 --ns_steps $steps \
      --save_dir $SAVE_DIR --tag_suffix "st${steps}" 2>&1 | tail -1
  done
done

echo "=== 2.1 ALL DONE ==="
