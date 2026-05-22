#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable_worldmodel}"
PROBE_TYPE="${PROBE_TYPE:-knn}"
PARALLEL_JOBS="${PARALLEL_JOBS:-5}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/experiments/results/probing_suite_$TIMESTAMP}"

mkdir -p "$RESULTS_DIR"

declare -a MODELS=(
  "metaworld_gated|$STABLEWM_HOME/metaworld_gated/metaworld_gated_epoch_2_object.ckpt"
  "metaworld_gated_masked|$STABLEWM_HOME/metaworld_gated_masked/metaworld_gated_masked_epoch_2_object.ckpt"
  "metaworld_pixels|$STABLEWM_HOME/metaworld_pixels/metaworld_pixels_epoch_2_object.ckpt"
  "metaworld_selfattention|$STABLEWM_HOME/metaworld_selfattention/metaworld_selfatt_epoch_2_object.ckpt"
  "metaworld_selfattention_masked|$STABLEWM_HOME/metaworld_selfattention_masked/metaworld_selfatt_masked_epoch_2_object.ckpt"
)

echo "Probing suite output directory: $RESULTS_DIR"
echo "Probe type: $PROBE_TYPE"
echo "Parallel jobs: $PARALLEL_JOBS"

run_one() {
  local model_label="$1"
  local checkpoint="$2"
  shift 2
  local output_json="$RESULTS_DIR/${model_label}_${PROBE_TYPE}.json"
  local output_log="$RESULTS_DIR/${model_label}_${PROBE_TYPE}.log"

  if [[ ! -f "$checkpoint" ]]; then
    echo "[$model_label] Missing checkpoint: $checkpoint" | tee "$output_log"
    return 1
  fi

  echo "[$model_label] Starting probe run"
  "$ROOT_DIR/job_dir/run_probe_experiments.sh" \
    "$checkpoint" \
    "$PROBE_TYPE" \
    --output "$output_json" \
    "$@" >"$output_log" 2>&1
  echo "[$model_label] Finished -> $output_json"
}

active_jobs=0
declare -a job_pids=()
declare -a job_labels=()
for entry in "${MODELS[@]}"; do
  IFS="|" read -r model_label checkpoint <<< "$entry"
  run_one "$model_label" "$checkpoint" "$@" &
  job_pids+=("$!")
  job_labels+=("$model_label")
  ((active_jobs+=1))

  if (( active_jobs >= PARALLEL_JOBS )); then
    set +e
    wait -n
    set -e
    ((active_jobs-=1))
  fi
done

failures=0
for idx in "${!job_pids[@]}"; do
  pid="${job_pids[$idx]}"
  label="${job_labels[$idx]}"
  set +e
  wait "$pid"
  status=$?
  set -e
  if (( status != 0 )); then
    echo "[$label] Failed with exit code $status"
    ((failures+=1))
  fi
done

echo ""
echo "Completed probing suite. Results:"
for entry in "${MODELS[@]}"; do
  IFS="|" read -r model_label _ <<< "$entry"
  echo "  - $RESULTS_DIR/${model_label}_${PROBE_TYPE}.json"
done

if (( failures != 0 )); then
  echo ""
  echo "Suite finished with $failures failed run(s). Check the per-model logs in $RESULTS_DIR."
  exit 1
fi
