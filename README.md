# Anomaly Detection in Multivariate Time Series with Diffusion Models

Master's thesis research codebase — **AGH University of Science and Technology**,  
*Artificial Intelligence and Data Analysis*.

This repository implements an experimental pipeline for **anomaly detection and localization** in multivariate time series using **Denoising Diffusion Probabilistic Models (DDPM)**. Models are trained on normal data only; at test time, anomalies are scored via **partial diffusion reconstruction error** and optional **denoising attribution maps (DAM)**.

> **Thesis topic (working title):** *Anomaly detection in multivariate time series using probabilistic diffusion models*

---

## Highlights

- **DDPM + U-Net 1D** with configurable partial diffusion steps (`k` forward / `k` backward)
- **DRE** (Diffusion Reconstruction Error) and **DAM** (Denoising Attribution Map) for detection and channel-level attribution
- **Baselines:** LSTM-AE, Transformer-AE, [TranAD](https://github.com/imperial-qore/TranAD) (wrapper)
- **Benchmarks:** SMD, Yahoo S5, Jena Weather, UCI Sensor Drift (+ custom CSV / Kaggle-labeled loader)
- **Evaluation:** point & range metrics, AUC-ROC/PR, simplified VUS-PR, F1 / POT / percentile thresholds
- **Training utilities:** EMA weights, DDIM fast inference, PLGrid SLURM scripts

---

## Repository layout

```
├── configs/              # YAML experiment configs (smd, yahoo, weather, …)
├── src/tsad/             # Core library
│   ├── data/             # Dataset loaders & sliding-window splits
│   ├── models/           # DDPM, U-Net, baselines (LSTM / Transformer / TranAD)
│   ├── scoring/          # DRE, DAM, reconstruction scores
│   └── eval/             # Metrics, threshold selection, evaluation runner
├── experiments/          # Training, evaluation, ablation, benchmarking
├── scripts/              # Dataset download scripts, PLGrid helpers
├── data/                 # Datasets (not tracked — download locally)
├── checkpoints/          # Saved models (not tracked)
└── results/              # Metrics & summaries (not tracked)
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- See [`requirements.txt`](requirements.txt)

```bash
pip install -r requirements.txt
```

---

## Quick start

### 1. Verify setup

```bash
python experiments/verify_setup.py
```

### 2. Download datasets

```bash
python scripts/download_smd.py
python scripts/download_yahoo.py
python scripts/download_weather.py
python scripts/download_sensor_drift.py
```

### 3. Train DDPM (SMD, CPU-friendly config)

```bash
python experiments/train_ddpm.py --config configs/smd.yaml --epochs 20
```

### 4. Compare models

```bash
python experiments/train_lstm_ae.py --config configs/smd.yaml --epochs 25
python experiments/train_transformer_ae.py --config configs/smd.yaml --epochs 25
python experiments/train_tranad.py --config configs/smd.yaml --epochs 25
python experiments/compare_all.py --config configs/smd.yaml
```

Results are written to `results/<dataset>/summary.csv`.

---

## Key experiments

| Script | Description |
|--------|-------------|
| `experiments/train_ddpm.py` | Train DDPM on normal windows only |
| `experiments/compare_all.py` | Compare DDPM vs. baselines → CSV |
| `experiments/ablate_k.py` | Ablation over diffusion step `k` |
| `experiments/run_benchmarks.py` | Multi-dataset train/eval pipeline |
| `experiments/plot_dam.py` | Visualize denoising attribution maps |
| `experiments/plot_channels.py` | Channel × time heatmaps (SMD) |
| `experiments/run_smd_machines.py` | Evaluate multiple SMD machines |

---

## Configuration

Experiments are driven by YAML files under `configs/`:

| Config | Purpose |
|--------|---------|
| `smd.yaml` | Fast CPU experiments (window limits, larger stride) |
| `smd_full.yaml` | Full SMD evaluation, EMA, no window caps |
| `plgrid_smd_full.yaml` | GPU / PLGrid: CUDA, EMA, DDIM |
| `yahoo.yaml`, `weather.yaml`, `sensor_drift.yaml` | Additional benchmarks |
| `kaggle_labeled.yaml` | Custom labeled CSV (place file under `data/kaggle/`) |

Important options:

```yaml
scoring:
  diffusion_steps: [25, 50, 75]      # partial diffusion start steps (DRE ensemble)
  reconstruct_method: ddpm           # ddpm | ddim

eval:
  threshold_method: f1                 # f1 | pot | percentile

train:
  ema:
    enabled: true
    decay: 0.999
```

---

## Method (short)

1. **Training** — DDPM learns to denoise normal multivariate windows.
2. **Inference** — add noise for `k` steps, reconstruct for `k` steps (partial diffusion).
3. **Scoring** — per-timestep reconstruction error → anomaly score; DAM aggregates errors across diffusion steps.
4. **Threshold** — tuned on validation (F1) or unsupervised (POT / percentile on normal train scores).

---

## Data notes

| Dataset | Source | Labels |
|---------|--------|--------|
| **SMD** | [Server Machine Dataset](https://github.com/NetManAIOps/OmniAnomaly) | Point & segment anomalies |
| **Yahoo S5** | Webscope / public mirrors | `is_anomaly` |
| **Weather** | Jena station (Autoformer bundle) | Unsupervised (no native labels) |
| **Sensor Drift** | [UCI Gas Sensor Array Drift](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations) | Batch-drift labels |

Large files under `data/`, `checkpoints/`, and `results/` are excluded from Git — regenerate locally with the download and training scripts.

---

## PLGrid (optional)

```bash
sbatch scripts/plgrid/submit_ddpm.slurm configs/plgrid_smd_full.yaml 60
sbatch scripts/plgrid/submit_benchmarks.slurm
```

Edit `scripts/plgrid/env.sh` for your cluster modules before submitting.

---

## Author

**Michał Burda** — [github.com/MichalBurda4](https://github.com/MichalBurda4)

---

## License

This repository is part of an academic master's thesis. Dataset licenses apply separately (SMD, Yahoo Webscope, UCI, etc.). Code may be shared for research purposes; contact the author for other uses.
