#!/usr/bin/env bash

set -Eeuo pipefail

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
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR"
DATASET_NAME="${DATASET_NAME:-metaworld_eval}"
NUM_RUNS="${NUM_RUNS:-3}"
BASE_SEED="${BASE_SEED:-42}"
PARALLEL_JOBS="${PARALLEL_JOBS:-1}"
export PARALLEL_JOBS
NUM_EVAL="${NUM_EVAL:-10}"
GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-25}"
EVAL_BUDGET="${EVAL_BUDGET:-50}"
HORIZON="${HORIZON:-25}"
RECEDING_HORIZON="${RECEDING_HORIZON:-5}"
ACTION_BLOCK="${ACTION_BLOCK:-1}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

REPORT_DIR="${REPORT_DIR:-$ROOT_DIR/documentation/planning_suite_$TIMESTAMP}"
mkdir -p "$REPORT_DIR"

RAW_JSONL="$REPORT_DIR/planning_runs.jsonl"
RAW_LOG="$REPORT_DIR/planning_runs.log"
REPORT_MD="$REPORT_DIR/planning_report.md"
LOCK_FILE="$REPORT_DIR/planning_suite.lock"

touch "$RAW_JSONL" "$RAW_LOG"

append_success_record() {
  RECORD_PATH="$1" \
  RECORD_TASK="$2" \
  RECORD_MODEL_LABEL="$3" \
  RECORD_POLICY_REF="$4" \
  RECORD_RUN_IDX="$5" \
  RECORD_SEED="$6" \
  RECORD_METRICS_JSON="$7" \
  "$PYTHON_BIN" -c '
import json
import os

record = {
    "task": os.environ["RECORD_TASK"],
    "model_label": os.environ["RECORD_MODEL_LABEL"],
    "policy_ref": os.environ["RECORD_POLICY_REF"],
    "run_idx": int(os.environ["RECORD_RUN_IDX"]),
    "seed": int(os.environ["RECORD_SEED"]),
    "status": "ok",
    "metrics": json.loads(os.environ["RECORD_METRICS_JSON"]),
}
with open(os.environ["RECORD_PATH"], "a", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\n")
'
}

append_failure_record() {
  RECORD_PATH="$1" \
  RECORD_TASK="$2" \
  RECORD_MODEL_LABEL="$3" \
  RECORD_POLICY_REF="$4" \
  RECORD_RUN_IDX="$5" \
  RECORD_SEED="$6" \
  RECORD_EXIT_CODE="$7" \
  "$PYTHON_BIN" -c '
import json
import os

record = {
    "task": os.environ["RECORD_TASK"],
    "model_label": os.environ["RECORD_MODEL_LABEL"],
    "policy_ref": os.environ["RECORD_POLICY_REF"],
    "run_idx": int(os.environ["RECORD_RUN_IDX"]),
    "seed": int(os.environ["RECORD_SEED"]),
    "status": "failed",
    "exit_code": int(os.environ["RECORD_EXIT_CODE"]),
}
with open(os.environ["RECORD_PATH"], "a", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\n")
'
}

render_report() {
  "$PYTHON_BIN" - <<'PY' "$RAW_JSONL" "$REPORT_MD" "$NUM_RUNS" "$DATASET_NAME" "$BASE_SEED"
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

raw_path = Path(sys.argv[1])
report_path = Path(sys.argv[2])
num_runs = int(sys.argv[3])
dataset_name = sys.argv[4]
base_seed = int(sys.argv[5])

records = []
with raw_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

grouped = defaultdict(list)
for record in records:
    grouped[(record["task"], record["model_label"])].append(record)

tasks = sorted({task for task, _ in grouped.keys()})
models = sorted({model for _, model in grouped.keys()})

lines = []
lines.append("# Planning Benchmark Report")
lines.append("")
lines.append(f"- Dataset: `{dataset_name}`")
lines.append(f"- Runs per task/model: `{num_runs}`")
lines.append(f"- Seeds: `{base_seed}` to `{base_seed + num_runs - 1}`")
lines.append(f"- Parallel jobs: `{os.environ.get('PARALLEL_JOBS', 'unknown')}`")
lines.append(f"- Raw records: `{raw_path}`")
lines.append("")

for task in tasks:
    lines.append(f"## {task}")
    lines.append("")
    lines.append("| Model | Success mean (%) | Success var | Latent distance mean | Latent distance var | Run details |")
    lines.append("|---|---:|---:|---:|---:|---|")

    for model in models:
        runs = sorted(grouped.get((task, model), []), key=lambda r: r["run_idx"])
        ok_runs = [r for r in runs if r["status"] == "ok"]

        if not ok_runs:
            failure_codes = ", ".join(str(r.get("exit_code", "?")) for r in runs) or "n/a"
            lines.append(f"| {model} | failed | failed | failed | failed | exit codes: {failure_codes} |")
            continue

        success_rates = np.array(
            [float(r["metrics"]["success_rate"]) for r in ok_runs], dtype=float
        )
        latent_means = np.array(
            [
                float(r["metrics"].get("final_latent_goal_distance_mean", np.nan))
                for r in ok_runs
            ],
            dtype=float,
        )

        success_mean = float(success_rates.mean())
        success_var = float(success_rates.var())

        valid_latent = latent_means[~np.isnan(latent_means)]
        if len(valid_latent) > 0:
            latent_mean = float(valid_latent.mean())
            latent_var = float(valid_latent.var())
            latent_mean_str = f"{latent_mean:.6f}"
            latent_var_str = f"{latent_var:.6f}"
        else:
            latent_mean_str = "n/a"
            latent_var_str = "n/a"

        details = []
        for run in runs:
            if run["status"] != "ok":
                details.append(f"run {run['run_idx']}: failed({run.get('exit_code', '?')})")
                continue
            metrics = run["metrics"]
            success = float(metrics["success_rate"])
            latent = metrics.get("final_latent_goal_distance_mean", None)
            if latent is None:
                details.append(f"run {run['run_idx']}: success={success:.1f}, latent=n/a")
            else:
                details.append(
                    f"run {run['run_idx']}: success={success:.1f}, latent={float(latent):.6f}"
                )

        lines.append(
            f"| {model} | {success_mean:.2f} | {success_var:.4f} | "
            f"{latent_mean_str} | {latent_var_str} | {'; '.join(details)} |"
        )

    lines.append("")

tmp_path = report_path.with_suffix(report_path.suffix + ".tmp")
tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
tmp_path.replace(report_path)
print(report_path)
PY
}

declare -a MODELS=(
  "metaworld_concatproject_2|metaworld_concatproject_2/metaworld_concatproject_2_epoch_3|"
  "metaworld_gated|metaworld_gated/metaworld_gated_epoch_3|"
  "metaworld_gated_masked|metaworld_gated_masked/metaworld_gated_masked_epoch_3|"
  "metaworld_pixels|metaworld_pixels/metaworld_pixels_epoch_3|"
  "metaworld_selfattention|metaworld_selfattention/metaworld_selfatt_epoch_3|"
  "metaworld_selfattention_masked|metaworld_selfattention_masked/metaworld_selfatt_masked_epoch_3|"
)

if [[ -n "${METAWORLD_TASKS:-}" ]]; then
  IFS=',' read -r -a TASKS <<< "$METAWORLD_TASKS"
else
  TASKS=("door-open-v3" "drawer-close-v3" "drawer-open-v3" "reach-v3")
fi

validate_model_checkpoints() {
  local missing=0
  local model_label
  local policy_ref
  local extra_override
  local ckpt_path

  for model_spec in "${MODELS[@]}"; do
    IFS='|' read -r model_label policy_ref extra_override <<< "$model_spec"
    if [[ "$policy_ref" == "random" ]]; then
      continue
    fi

    ckpt_path="$STABLEWM_HOME/${policy_ref}_object.ckpt"
    if [[ ! -f "$ckpt_path" ]]; then
      echo "Missing checkpoint for $model_label: $ckpt_path" >&2
      missing=1
    fi
  done

  if (( missing != 0 )); then
    exit 1
  fi
}

validate_model_checkpoints

echo "Planning suite report directory: $REPORT_DIR"
echo "Tasks: ${TASKS[*]}"
echo "Raw run records: $RAW_JSONL"
echo "Parallel jobs: $PARALLEL_JOBS"

run_one() {
  local task="$1"
  local model_label="$2"
  local policy_ref="$3"
  local extra_override="$4"
  local run_idx="$5"
  local seed="$6"

  local -a cmd=(
    "$PYTHON_BIN" eval.py
    --config-name=metaworld
    "policy=$policy_ref"
    "eval.dataset_name=$DATASET_NAME"
    "world.metaworld_env_name=$task"
    "seed=$seed"
    "eval.num_eval=$NUM_EVAL"
    "eval.goal_offset_steps=$GOAL_OFFSET_STEPS"
    "eval.eval_budget=$EVAL_BUDGET"
    "plan_config.horizon=$HORIZON"
    "plan_config.receding_horizon=$RECEDING_HORIZON"
    "plan_config.action_block=$ACTION_BLOCK"
    "output.filename=planning_suite_eval_${TIMESTAMP}.txt"
  )
  if [[ -n "$extra_override" ]]; then
    cmd+=("$extra_override")
  fi

  local run_log="$REPORT_DIR/${task}__${model_label}__run_${run_idx}.log"
  local cmd_line
  printf -v cmd_line '%q ' "${cmd[@]}"

  {
    echo "=== TASK=$task MODEL=$model_label RUN=$run_idx SEED=$seed ==="
    echo "Command: $cmd_line"
  } | tee -a "$RAW_LOG" "$run_log"

  set +e
  "${cmd[@]}" 2>&1 | tee -a "$RAW_LOG" "$run_log"
  local status=${PIPESTATUS[0]}
  set -e

  local metrics_json
  metrics_json="$(awk '/^METRICS_JSON=/{sub(/^METRICS_JSON=/,""); print}' "$run_log" | tail -n 1)"

  (
    flock -x 9
    if [[ $status -eq 0 && -n "$metrics_json" ]]; then
      append_success_record \
        "$RAW_JSONL" \
        "$task" \
        "$model_label" \
        "$policy_ref" \
        "$run_idx" \
        "$seed" \
        "$metrics_json"
    else
      local failure_status="$status"
      if [[ -z "$metrics_json" && "$failure_status" -eq 0 ]]; then
        failure_status=1
      fi

      append_failure_record \
        "$RAW_JSONL" \
        "$task" \
        "$model_label" \
        "$policy_ref" \
        "$run_idx" \
        "$seed" \
        "$failure_status"
    fi
    render_report >/dev/null
  ) 9>"$LOCK_FILE"

  if [[ $status -ne 0 ]]; then
    echo "Planning run failed with exit code $status. See $run_log" >&2
    return "$status"
  fi

  if [[ -z "$metrics_json" ]]; then
    echo "Planning run finished without METRICS_JSON. See $run_log" >&2
    return 1
  fi
}

render_report >/dev/null

stop_running_jobs() {
  jobs -pr | xargs -r kill 2>/dev/null || true
  wait 2>/dev/null || true
}

wait_for_one_job() {
  local status

  set +e
  wait -n
  status=$?
  set -e

  running_jobs=$((running_jobs - 1))
  if (( status != 0 )); then
    stop_running_jobs
    echo "Planning suite stopped after the first failed run. See $RAW_LOG" >&2
    exit "$status"
  fi
}

trap 'stop_running_jobs' INT TERM

running_jobs=0
for task in "${TASKS[@]}"; do
  for model_spec in "${MODELS[@]}"; do
    IFS='|' read -r model_label policy_ref extra_override <<< "$model_spec"

    for ((run_idx=1; run_idx<=NUM_RUNS; run_idx++)); do
      seed=$((BASE_SEED + run_idx - 1))
      if (( PARALLEL_JOBS <= 1 )); then
        run_one "$task" "$model_label" "$policy_ref" "$extra_override" "$run_idx" "$seed"
      else
        run_one "$task" "$model_label" "$policy_ref" "$extra_override" "$run_idx" "$seed" &
        running_jobs=$((running_jobs + 1))
        if (( running_jobs >= PARALLEL_JOBS )); then
          wait_for_one_job
        fi
      fi
    done
  done
done

while (( running_jobs > 0 )); do
  wait_for_one_job
done

render_report >/dev/null

echo "Planning report written to $REPORT_MD"
