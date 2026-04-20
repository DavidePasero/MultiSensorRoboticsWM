#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 2 ]]; then
  cat <<'EOF'
Usage:
  job_dir/run_probe_experiments.sh <checkpoint> <probe_type> [extra probe args...]

This launcher always runs all available probe tasks:
  - ee_position
  - contact_no_contact
  - object_distance

Examples:
  job_dir/run_probe_experiments.sh ~/.stable_worldmodel/lewm_epoch_3_object.ckpt linear --device cuda
  job_dir/run_probe_experiments.sh ~/.stable_worldmodel/metaworld_selfattention/lewm_epoch_3_object.ckpt knn --knn-k 32 --device cuda
EOF
  exit 1
fi

CHECKPOINT="$1"
PROBE_TYPE="$2"
shift 2

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable_worldmodel}"

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" experiments/probe_experiments.py \
  "${CHECKPOINT}" \
  --experiments ee_position contact_no_contact object_distance \
  --probe-type "${PROBE_TYPE}" \
  "$@"
