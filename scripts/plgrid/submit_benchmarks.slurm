#!/bin/bash
#SBATCH --job-name=tsad_bench
#SBATCH --output=logs/plgrid_bench_%j.out
#SBATCH --error=logs/plgrid_bench_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=plgrid-gpu
#
# Submit: sbatch scripts/plgrid/submit_benchmarks.slurm
# Optional env: BENCHMARK_CONFIGS, BENCHMARK_MODELS, BENCHMARK_EPOCHS

set -euo pipefail

CONFIGS="${BENCHMARK_CONFIGS:-configs/smd.yaml,configs/yahoo.yaml,configs/sensor_drift.yaml}"
MODELS="${BENCHMARK_MODELS:-ddpm,lstm_ae,transformer_ae,tranad}"
EPOCHS="${BENCHMARK_EPOCHS:-30}"

source scripts/plgrid/env.sh
mkdir -p logs

python experiments/run_benchmarks.py \
  --train \
  --epochs "$EPOCHS" \
  --configs "$CONFIGS" \
  --models "$MODELS"

echo "Saved: results/master_summary.csv"
