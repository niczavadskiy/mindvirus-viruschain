#!/usr/bin/env bash
# GPU instance / notebook host: install deps and run the locked HF mini smoke.
# Usage (from experiments/evolution_epochs):
#   bash scripts/run_instance_hf_mini.sh
#   bash scripts/run_instance_hf_mini.sh configs/smoke_hf_mini.yaml
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/smoke_hf_mini_instruct.yaml}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN not set. Create $ROOT/.env from .env.example" >&2
fi

python -m pip install -q -r requirements-hf.txt
python -m pip install -q -e .
python -m pip uninstall -y torchaudio torchvision sentence-transformers >/dev/null 2>&1 || true

echo "cuda=$(python -c 'import torch; print(torch.cuda.is_available())')"
echo "config=$CONFIG"
exec python -m steerchain --config "$CONFIG"
