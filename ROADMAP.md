# Kolejne kroki — magisterka



## Zrobione

- [x] Prototyp DDPM + DRE/DAM

- [x] Pipeline train / eval / LSTM baseline

- [x] SMD, Weather, Yahoo, SensorDrift loaders

- [x] ablate_k, compare_all, plot_channels, run_smd_machines

- [x] **POT / percentile / F1 threshold** (`eval.threshold_method`)

- [x] **Range F1 + VUS-PR (simplified)**

- [x] **`run_benchmarks.py`** → `results/master_summary.csv`

- [x] **Transformer-AE baseline**

- [x] **EMA weights** + `configs/smd_full.yaml`

- [x] **TranAD wrapper**

- [x] **PLGrid** — `scripts/plgrid/*.slurm`, `configs/plgrid_smd_full.yaml`

- [x] **SensorDrift drift labels** (`label_mode: drift`, `train_batches`)

- [x] **DDIM fast inference** (`scoring.reconstruct_method: ddim`)

- [x] **Kaggle labeled CSV loader** (`configs/kaggle_labeled.yaml`)



## Teraz — uruchom



```powershell

# Lokalnie — porównanie wszystkich modeli

python experiments/compare_all.py --config configs/smd.yaml



# DDIM zamiast pełnego DDPM (w configu: scoring.reconstruct_method: ddim)

python experiments/compare_all.py --config configs/smd.yaml



# SensorDrift z etykietami batch-drift

python scripts/download_sensor_drift.py

python experiments/train_ddpm.py --config configs/sensor_drift.yaml



# Kaggle labeled CSV (po wrzuceniu pliku do data/kaggle/)

python scripts/download_kaggle_labeled.py

python experiments/train_ddpm.py --config configs/kaggle_labeled.yaml



# PLGrid (na login node klastra)

sbatch scripts/plgrid/submit_ddpm.slurm configs/plgrid_smd_full.yaml 60

sbatch scripts/plgrid/submit_benchmarks.slurm

```



## Opcjonalnie później

- [ ] Score-based SDE (drugi model dyfuzyjny z opisu promotora)

- [ ] Notebook case studies / wykresy do rozdziału wyników

- [ ] Pisanie pracy (rozdziały równolegle z eksperymentami)

