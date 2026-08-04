#!/bin/bash
# Shared PLGrid environment — edit module names for your cluster site.
# Docs: https://docs.plgrid.pl/

set -euo pipefail

PROJECT_ROOT="${PLGRID_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

# Athena / Ares examples — uncomment what matches your site:
# module load Python/3.10.4-GCCcore-11.3.0
# module load CUDA/11.7.0

if [[ -d ".venv" ]]; then
  source .venv/bin/activate
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export PLGRID_PROJECT_ROOT="$PROJECT_ROOT"

echo "Project: $PROJECT_ROOT"
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
