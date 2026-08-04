#!/bin/bash
#SBATCH --job-name=tsad_ddpm
#SBATCH --output=logs/plgrid_ddpm_%j.out
#SBATCH --error=logs/plgrid_ddpm_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=plgrid-gpu
#
# Submit: sbatch scripts/plgrid/submit_ddpm.slurm [config] [epochs]
# Example: sbatch scripts/plgrid/submit_ddpm.slurm configs/plgrid_smd_full.yaml 60

#SBATCH --chdir=.

set -euo pipefail

CONFIG="${1:-configs/plgrid_smd_full.yaml}"
EPOCHS="${2:-60}"

source scripts/plgrid/env.sh
mkdir -p logs

echo "Training DDPM: config=$CONFIG epochs=$EPOCHS"
python experiments/train_ddpm.py --config "$CONFIG" --epochs "$EPOCHS"

python experiments/compare_all.py --config "$CONFIG"

echo "Done. Results in results/"
