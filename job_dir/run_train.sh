#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  job_dir/run_train.sh <hydra overrides...>

Examples:
  job_dir/run_train.sh data=metaworld data.dataset.name=metaworld_v2 obs_encoder=multimodal
  job_dir/run_train.sh data=metaworld obs_encoder=multimodal obs_encoder.fusion.type=selfattention trainer.max_epochs=3
EOF
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable_worldmodel}"

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" train.py "$@"
