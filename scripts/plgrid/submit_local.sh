#!/usr/bin/env bash
# Local helper — prints sbatch commands (run on PLGrid login node).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs

echo "=== PLGrid job submission ==="
echo ""
echo "1) Full DDPM training (GPU):"
echo "   sbatch scripts/plgrid/submit_ddpm.slurm configs/plgrid_smd_full.yaml 60"
echo ""
echo "2) All benchmarks train+eval:"
echo "   sbatch scripts/plgrid/submit_benchmarks.slurm"
echo ""
echo "3) Custom benchmarks:"
echo "   BENCHMARK_CONFIGS=configs/smd.yaml BENCHMARK_EPOCHS=25 sbatch scripts/plgrid/submit_benchmarks.slurm"
echo ""
echo "Before first run on cluster:"
echo "   module load Python/3.x CUDA/11.x   # site-specific"
echo "   python -m venv .venv && source .venv/bin/activate"
echo "   pip install -r requirements.txt"
echo "   Edit scripts/plgrid/env.sh with your module names"
