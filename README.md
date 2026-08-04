# Magisterka: DDPM + anomalie w szeregach czasowych

Projekt badawczy — od prototypu (`prototype/`) do pełnego pipeline'u (`src/tsad/`).

## Struktura

```
magisterka/
├── configs/default.yaml      # konfiguracja eksperymentów
├── src/tsad/                 # biblioteka
│   ├── data/                 # strumień syntetyczny, okna, CSV
│   ├── models/               # DDPM (U-Net), LSTM-AE
│   ├── scoring/              # DRE + DAM
│   └── eval/                 # F1, AUC, attribution recall
├── experiments/
│   ├── train_ddpm.py
│   ├── train_lstm_ae.py
│   ├── evaluate.py
│   ├── plot_dam.py
│   └── run_all.py
├── prototype/                # v0 — prosty demo (zostawiony)
├── checkpoints/
├── results/
└── outputs/
```

## Nowe skrypty

| Skrypt | Opis |
|--------|------|
| `experiments/ablate_k.py` | Ablacja kroków dyfuzji (`--quick` na szybki test) |
| `experiments/compare_all.py` | DDPM vs LSTM → `summary.csv` |
| `experiments/plot_channels.py` | Heatmapa kanał × czas (SMD) |
| `experiments/run_smd_machines.py` | Wiele maszyn SMD |
| `scripts/download_*.py` | Yahoo, Weather, SensorDrift, Kaggle labeled |
| `scripts/plgrid/` | SLURM joby na PLGrid (GPU) |

## Konfiguracje

- `configs/smd.yaml` — szybki test CPU
- `configs/smd_full.yaml` — pełny test bez limitów okien
- `configs/plgrid_smd_full.yaml` — smd_full + CUDA + DDIM + EMA
- `configs/yahoo.yaml`, `weather.yaml`, `sensor_drift.yaml`
- `configs/kaggle_labeled.yaml` — labeled CSV (Kaggle / własny)

```powershell
cd magisterka
pip install -r requirements.txt
python experiments/run_all.py
```

Krok po kroku:

```powershell
python experiments/train_ddpm.py --epochs 30
python experiments/train_lstm_ae.py --epochs 30
python experiments/evaluate.py
python experiments/plot_dam.py
```

## Co nowego względem `prototype/`

| Element | Opis |
|---------|------|
| **Długi strumień** | Train/val/test z etykietami punktowymi |
| **Okna** | Sliding window; trening tylko na normalnych |
| **DRE** | Ensemble błędu rekonstrukcji po kilku `start_t` |
| **DAM** | Heatmapa czas × krok dyfuzji |
| **LSTM-AE** | Baseline do porównania w `results/metrics.json` |
| **Metryki** | Point F1, AUC-ROC/PR, window F1, attribution recall |
| **Early stopping** | Na val loss |

## Własne dane (CSV)

W `configs/default.yaml`:

```yaml
data:
  name: csv
  csv_path: data/moja_seria.csv
```

Kolumny: `value` (wymagane), `label` (0/1, opcjonalne). Wymaga `pandas` (`pip install pandas`).

## Następne kroki

- Pełny trening na PLGrid: `sbatch scripts/plgrid/submit_ddpm.slurm`
- Porównanie baseline'ów: `python experiments/compare_all.py --config configs/smd.yaml`
- DDIM inference: `scoring.reconstruct_method: ddim` w configu
- SensorDrift z etykietami drift: `configs/sensor_drift.yaml` (`label_mode: drift`)
