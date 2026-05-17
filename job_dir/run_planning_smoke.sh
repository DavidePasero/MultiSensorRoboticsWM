#!/usr/bin/env bash

set -u -o pipefail

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
DATASET_NAME="${DATASET_NAME:-metaworld_eval_converted}"
TASK="${TASK:-reach-v3}"
SEED="${SEED:-42}"
NUM_EVAL="${NUM_EVAL:-2}"
GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-5}"
EVAL_BUDGET="${EVAL_BUDGET:-5}"
HORIZON="${HORIZON:-5}"
RECEDING_HORIZON="${RECEDING_HORIZON:-1}"
ACTION_BLOCK="${ACTION_BLOCK:-1}"
SOLVER_NUM_SAMPLES="${SOLVER_NUM_SAMPLES:-32}"
SOLVER_TOPK="${SOLVER_TOPK:-8}"
SOLVER_N_STEPS="${SOLVER_N_STEPS:-3}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

REPORT_DIR="${REPORT_DIR:-$ROOT_DIR/documentation/planning_smoke_$TIMESTAMP}"
mkdir -p "$REPORT_DIR"

LOG_PATH="$REPORT_DIR/planning_smoke.log"
SUMMARY_PATH="$REPORT_DIR/planning_smoke_summary.md"

declare -a MODELS=(
  "metaworld_force_torque|metaworld_force_torque"
  "metaworld_gated|metaworld_gated"
  "metaworld_pixels|metaworld_pixels"
  "metaworld_selfattention|metaworld_selfattention"
  "metaworld_selfattention_masked|metaworld_selfattention_masked"
)

echo "# Planning Smoke Test" > "$SUMMARY_PATH"
echo "" >> "$SUMMARY_PATH"
echo "- Dataset: \`$DATASET_NAME\`" >> "$SUMMARY_PATH"
echo "- Task: \`$TASK\`" >> "$SUMMARY_PATH"
echo "- Seed: \`$SEED\`" >> "$SUMMARY_PATH"
echo "- Num eval episodes: \`$NUM_EVAL\`" >> "$SUMMARY_PATH"
echo "- Eval budget: \`$EVAL_BUDGET\`" >> "$SUMMARY_PATH"
echo "- Solver samples: \`$SOLVER_NUM_SAMPLES\`" >> "$SUMMARY_PATH"
echo "- Solver topk: \`$SOLVER_TOPK\`" >> "$SUMMARY_PATH"
echo "- Solver steps: \`$SOLVER_N_STEPS\`" >> "$SUMMARY_PATH"
echo "" >> "$SUMMARY_PATH"
echo "| Model | Status | Success rate | Notes |" >> "$SUMMARY_PATH"
echo "|---|---|---:|---|" >> "$SUMMARY_PATH"

echo "Smoke-test report directory: $REPORT_DIR"
echo "Task: $TASK"
echo "Dataset: $DATASET_NAME"

for model_spec in "${MODELS[@]}"; do
  IFS='|' read -r model_label policy_ref <<< "$model_spec"

  cmd=(
    "$PYTHON_BIN" eval.py
    --config-name=metaworld
    "policy=$policy_ref"
    "eval.dataset_name=$DATASET_NAME"
    "world.metaworld_env_name=$TASK"
    "seed=$SEED"
    "eval.num_eval=$NUM_EVAL"
    "eval.goal_offset_steps=$GOAL_OFFSET_STEPS"
    "eval.eval_budget=$EVAL_BUDGET"
    "plan_config.horizon=$HORIZON"
    "plan_config.receding_horizon=$RECEDING_HORIZON"
    "plan_config.action_block=$ACTION_BLOCK"
    "solver.num_samples=$SOLVER_NUM_SAMPLES"
    "solver.topk=$SOLVER_TOPK"
    "solver.n_steps=$SOLVER_N_STEPS"
    "output.filename=planning_smoke_eval_${TIMESTAMP}.txt"
  )

  echo "=== MODEL=$model_label TASK=$TASK SEED=$SEED ===" | tee -a "$LOG_PATH"
  output="$("${cmd[@]}" 2>&1)"
  status=$?
  printf '%s\n' "$output" >> "$LOG_PATH"

  metrics_json="$(printf '%s\n' "$output" | awk '/^METRICS_JSON=/{sub(/^METRICS_JSON=/,""); print}' | tail -n 1)"

  if [[ $status -eq 0 && -n "$metrics_json" ]]; then
    success_rate="$("$PYTHON_BIN" -c 'import json,sys; print(json.loads(sys.argv[1])["success_rate"])' "$metrics_json")"
    echo "| $model_label | ok | $success_rate | see \`planning_smoke.log\` |" >> "$SUMMARY_PATH"
  else
    short_error="$(printf '%s\n' "$output" | tail -n 1 | tr '|' '/' | sed 's/`//g')"
    echo "| $model_label | failed | n/a | exit=$status; $short_error |" >> "$SUMMARY_PATH"
  fi
done

echo ""
echo "Smoke-test summary written to $SUMMARY_PATH"
