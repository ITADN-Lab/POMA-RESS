#!/bin/bash
# Sync all experiment results from 12号, 14号, and copy local 13号 results to paper repo
# Run from this script's directory.

set -e
LOCAL_BASE=/home/hadoop/workstation/md/LafTJU-TII/experiments/results_aggregated
mkdir -p $LOCAL_BASE/fair12 $LOCAL_BASE/spectral13 $LOCAL_BASE/secom14

echo "=== Pulling 12号 fair-baseline results ==="
rsync -az -e "ssh -p 1053" hadoop@127.0.0.1:~/laftju_tii_exp/results_fair12/ $LOCAL_BASE/fair12/

echo "=== Pulling 14号 SECOM results ==="
rsync -az -e "ssh -p 1056" hadoop@127.0.0.1:~/laftju_tii_exp/results_secom14/ $LOCAL_BASE/secom14/

echo "=== Linking local 13号 spectral results ==="
rsync -az /home/hadoop/workstation/md/LafTJU-TII/experiments/results_spectral/ $LOCAL_BASE/spectral13/

echo "=== Counts ==="
echo "12号 fair: $(ls $LOCAL_BASE/fair12/*.json 2>/dev/null | wc -l) files"
echo "13本机 spectral: $(ls $LOCAL_BASE/spectral13/*.json 2>/dev/null | wc -l) files"
echo "14号 SECOM: $(ls $LOCAL_BASE/secom14/*.json 2>/dev/null | wc -l) files"
