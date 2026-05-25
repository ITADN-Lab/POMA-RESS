#!/bin/bash
# Run industrial experiments for LafTJU-TII paper
# Usage: bash run_industrial.sh [cwru|cmapss|all]

set -e
cd "$(dirname "$0")"

OPTIMIZERS="AdamW LAKTJU_NS LAKTJU_Lite"
SEEDS="42 123 456 789 2024"
NPROC_PER_NODE=1

run_cwru() {
    echo "=== CWRU Bearing Fault Diagnosis ==="
    for opt in $OPTIMIZERS; do
        for seed in $SEEDS; do
            echo "Running: CWRU optimizer=$opt seed=$seed"
            python train_cwru_fault.py \
                --data_dir ../data/cwru \
                --optimizer $opt \
                --seed $seed \
                --epochs 100 \
                --batch_size 64 \
                --save_dir ../results \
                2>&1 | tee ../results/log_cwru_${opt}_seed${seed}.txt
        done
    done
}

run_cmapss() {
    echo "=== NASA C-MAPSS RUL Prediction ==="
    for subset in FD001 FD002 FD003 FD004; do
        for opt in $OPTIMIZERS; do
            for seed in $SEEDS; do
                echo "Running: C-MAPSS $subset optimizer=$opt seed=$seed"
                python train_cmapss_rul.py \
                    --data_dir ../data/cmapss \
                    --subset $subset \
                    --optimizer $opt \
                    --seed $seed \
                    --epochs 100 \
                    --batch_size 256 \
                    --save_dir ../results \
                    2>&1 | tee ../results/log_cmapss_${subset}_${opt}_seed${seed}.txt
            done
        done
    done
}

case "${1:-all}" in
    cwru)  run_cwru ;;
    cmapss) run_cmapss ;;
    all)   run_cwru; run_cmapss ;;
    *)     echo "Usage: $0 [cwru|cmapss|all]"; exit 1 ;;
esac

echo "All experiments complete."
