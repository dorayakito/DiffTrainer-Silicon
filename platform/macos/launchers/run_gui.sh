#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$repo_root"
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec python difftrainer.py "$@"
