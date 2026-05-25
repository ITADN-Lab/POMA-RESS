# POMA-RESS

Per-seed results, training and aggregator scripts, and figure-generation
code for the manuscript

> **Auditing Optimizer Claims for Industrial Recurrent Time-Series Training:
> A Leak-Free Equal-Budget Protocol with a Multi-Optimizer Case-Study Panel**
> *(Reliability Engineering & System Safety, under review, 2026.)*

The audit evaluates a four-optimizer panel — **POMA** (Periodic Orthogonal
Momentum for AdamW), **Adan**, **RAdam**, **Lion** — across three industrial
prognostic asset classes: NASA C-MAPSS turbofans, PHM 2012 PRONOSTIA FEMTO
bearings (5 leave-bearings-out partitions × 20 seeds), and NASA Ames PCoE
Li-ion batteries.

## Repository layout

```
scripts/                          training, audit, and aggregator scripts
  train_cmapss_leakfree.py        C-MAPSS leak-free trainer (FD001-FD004)
  train_femto_leakfree.py         FEMTO bearings leak-free trainer
  train_battery_leakfree.py       NASA Li-ion battery cross-domain trainer
  run_femto_multipart_20seed.py   5-partition × 20-seed FEMTO driver
  run_ims_audit.py                NASA IMS bearings 20-seed verification
  run_adan_audit.py               Adan-specific audit driver
  optimizers_extra.py             POMA, Adan, RAdam, Lion implementations
  aggregate_*.py                  per-seed JSON → table aggregators
  gen_*.py                        figure-generation scripts

results/
  results_femto/                  FEMTO summaries + 5-partition analysis
  results_battery/                Battery cross-domain summary
  results_ims/                    IMS bearings 20-seed summary
```

## Reproducing the main-paper tables

The audit uses one **equal-budget tuning grid** per optimizer
(`β₁ × GC × LR`) and one **leak-free leave-units-out** evaluation
protocol. All randomness is controlled by `--seed`.

```bash
# C-MAPSS FD001-FD004, leak-free, 20 seeds each
python scripts/train_cmapss_leakfree.py --subset FD002 --optimizer poma --seed 42
python scripts/aggregate_leakfree.py results/results_cmapss/

# FEMTO bearings, 5 partitions × 20 seeds
python scripts/run_femto_multipart_20seed.py --optimizer poma
python scripts/aggregate_femto_multipart.py

# NASA Li-ion battery cross-domain
python scripts/train_battery_leakfree.py --optimizer poma --seed 42
```

## Datasets

| Dataset | Source | License |
|---|---|---|
| NASA C-MAPSS | NASA Prognostics Data Repository | Public |
| PHM 2012 IEEE PRONOSTIA FEMTO | FEMTO-ST Institute | Public |
| NASA Ames PCoE Li-ion battery | NASA Prognostics Data Repository | Public |

The datasets are **not** redistributed here; download from the original
repositories using the bibliographic references in the manuscript.

## Computational footprint

Single-RTX-5090 audit budget: **≈ 90 GPU-h** per C-MAPSS subset, **≈ 60
GPU-h** per FEMTO partition × 20-seed sweep, **≈ 8 GPU-h** for the battery
cross-domain audit. The full panel sweep used **≈ 5,100 paired training
runs** on three RTX 5090 nodes.

## Software environment

- Python 3.11, PyTorch 2.9.1 + CUDA 12.8
- `transformers==4.57.6` (pinned; 5.x breaks per-tensor momentum state)
- See `scripts/run_*.sh` for exact CLI invocations.

## Citation

```bibtex
@article{poma_ress_2026,
  title   = {Auditing Optimizer Claims for Industrial Recurrent Time-Series
             Training: A Leak-Free Equal-Budget Protocol with a
             Multi-Optimizer Case-Study Panel},
  author  = {Lu, Teng-Chi and Lv, Xian-Long and Wang, Zhi-Yuan and Zeng, Lin},
  journal = {Reliability Engineering \& System Safety},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT (see `LICENSE`). The audit protocol is the deliverable — please cite
the manuscript when reusing the scripts or per-seed artefacts.
