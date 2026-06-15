#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=PLANNING
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=out_job_dir/PLANNING_%A_%a.out

set -euo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export STABLEWM_HOME="${STABLEWM_HOME:-/home/dpasero/project_space}"

if [[ -z "${PROJECT_DIR:-}" ]]; then
  if [[ -f /home/dpasero/MultiSensorRoboticsWM/eval.py ]]; then
    PROJECT_DIR="/home/dpasero/MultiSensorRoboticsWM"
  else
    SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
    PROJECT_DIR="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
  fi
fi
cd "$PROJECT_DIR"

if [[ ! -f eval.py ]]; then
  echo "Could not find eval.py in PROJECT_DIR=$PROJECT_DIR." >&2
  exit 1
fi
source .venv/bin/activate

EXPERIMENT_NAME="${EXPERIMENT_NAME:-manual_planning}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$STABLEWM_HOME/documentation/planning_corrected_eval70}"
OUTPUT_DIR_EXPLICIT="${OUTPUT_DIR+x}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$EXPERIMENT_NAME}"

MODEL_RUN="${MODEL_RUN:-random}"
DATASET_NAME="${DATASET_NAME:-metaworld_eval_button_press}"
TASK_LIST="${TASK_LIST:-button-press-v3}"
SEED_LIST="${SEED_LIST:-42 43 44}"

WORLD_HISTORY_SIZE="${WORLD_HISTORY_SIZE:-3}"
EVAL_NUM="${EVAL_NUM:-10}"
GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-20}"
EVAL_BUDGET="${EVAL_BUDGET:-70}"
HORIZON="${HORIZON:-15}"
RECEDING_HORIZON="${RECEDING_HORIZON:-5}"
ACTION_BLOCK="${ACTION_BLOCK:-1}"
WARM_START="${WARM_START:-true}"

SOLVER_VAR_SCALE="${SOLVER_VAR_SCALE:-0.3}"
CEM_NUM_SAMPLES="${CEM_NUM_SAMPLES:-300}"
CEM_TOPK="${CEM_TOPK:-30}"
CEM_STEPS="${CEM_STEPS:-30}"

CACHE_ALL_LOADED="${CACHE_ALL_LOADED:-true}"
SAVE_VIDEO="${SAVE_VIDEO:-true}"

# Keep these off for the corrected thesis rerun unless explicitly overridden.
CLAMP_ACTION_CANDIDATES="${CLAMP_ACTION_CANDIDATES:-false}"
FIRST_ACTION_DELTA_LIMIT="${FIRST_ACTION_DELTA_LIMIT:-null}"
ACTION_DELTA_LIMIT="${ACTION_DELTA_LIMIT:-null}"
EXECUTION_ACTION_DELTA_LIMIT="${EXECUTION_ACTION_DELTA_LIMIT:-null}"
EXECUTION_ACTION_NORM_LIMIT="${EXECUTION_ACTION_NORM_LIMIT:-null}"
ACTION_NORM_WEIGHT="${ACTION_NORM_WEIGHT:-0.0}"
ACTION_DELTA_WEIGHT="${ACTION_DELTA_WEIGHT:-0.0}"
FIRST_ACTION_DELTA_WEIGHT="${FIRST_ACTION_DELTA_WEIGHT:-0.0}"

MODALITY_SUBSTITUTION="${MODALITY_SUBSTITUTION:-impute}"
KEEP_MODALITIES_CSV="${KEEP_MODALITIES_CSV:-}"

BLUR="${BLUR:-false}"
BLUR_PROBABILITY="${BLUR_PROBABILITY:-1.0}"
BLUR_KERNEL_SIZE="${BLUR_KERNEL_SIZE:-5}"
BLUR_SIGMA_MIN="${BLUR_SIGMA_MIN:-1.0}"
BLUR_SIGMA_MAX="${BLUR_SIGMA_MAX:-1.0}"

if [[ -n "${PLANNING_MANIFEST:-}" ]]; then
  if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "PLANNING_MANIFEST is set, but SLURM_ARRAY_TASK_ID is empty." >&2
    exit 1
  fi
  if [[ ! -f "$PLANNING_MANIFEST" ]]; then
    echo "Planning manifest not found: $PLANNING_MANIFEST" >&2
    exit 1
  fi

  manifest_line="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PLANNING_MANIFEST")"
  if [[ -z "$manifest_line" ]]; then
    echo "No manifest entry for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID." >&2
    exit 1
  fi

  IFS=$'\t' read -r \
    EXPERIMENT_NAME \
    MODEL_RUN \
    DATASET_NAME \
    TASK_LIST \
    KEEP_MODALITIES_CSV \
    MODALITY_SUBSTITUTION \
    BLUR \
    BLUR_KERNEL_SIZE \
    BLUR_SIGMA \
    <<< "$manifest_line"

  BLUR_SIGMA_MIN="$BLUR_SIGMA"
  BLUR_SIGMA_MAX="$BLUR_SIGMA"
  if [[ -z "$OUTPUT_DIR_EXPLICIT" ]]; then
    OUTPUT_DIR="$OUTPUT_ROOT/$EXPERIMENT_NAME"
  fi

  echo "Loaded planning manifest entry $SLURM_ARRAY_TASK_ID from $PLANNING_MANIFEST"
fi

mkdir -p "$OUTPUT_DIR"

ALL_MODALITIES=(
  "pixels"
  "depth"
  "tactile"
  "proprio"
  "force_torque"
)

contains_modality() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

join_hydra_list() {
  local IFS=,
  echo "[$*]"
}

KEEP_MODALITIES=()
if [[ -n "$KEEP_MODALITIES_CSV" && "$KEEP_MODALITIES_CSV" != "all" && "$KEEP_MODALITIES_CSV" != "none" ]]; then
  normalized_keep="${KEEP_MODALITIES_CSV//,/ }"
  read -r -a KEEP_MODALITIES <<< "$normalized_keep"
fi

DROP_MODALITIES=()
case "$MODALITY_SUBSTITUTION" in
  impute|zero)
    ;;
  *)
    echo "MODALITY_SUBSTITUTION must be 'impute' or 'zero' (got '$MODALITY_SUBSTITUTION')." >&2
    exit 1
    ;;
esac

if [[ "${#KEEP_MODALITIES[@]}" -gt 0 ]]; then
  for modality in "${ALL_MODALITIES[@]}"; do
    if ! contains_modality "$modality" "${KEEP_MODALITIES[@]}"; then
      DROP_MODALITIES+=("$modality")
    fi
  done
fi

DROP_MODALITIES_OVERRIDE=()
if [[ "${#DROP_MODALITIES[@]}" -gt 0 ]]; then
  DROP_MODALITIES_OVERRIDE=(
    "eval.drop_modalities=$(join_hydra_list "${DROP_MODALITIES[@]}")"
    "eval.modality_substitution=$MODALITY_SUBSTITUTION"
  )
  echo "Keeping modalities: $(join_hydra_list "${KEEP_MODALITIES[@]}")"
  echo "Substituted modalities: $(join_hydra_list "${DROP_MODALITIES[@]}")"
  echo "Modality substitution: $MODALITY_SUBSTITUTION"
else
  echo "Keeping all modalities."
fi

BLUR_OVERRIDES=()
if [[ "$BLUR" == "true" ]]; then
  BLUR_OVERRIDES=(
    "eval.pixels_gaussian_blur.enabled=true"
    "eval.pixels_gaussian_blur.probability=$BLUR_PROBABILITY"
    "eval.pixels_gaussian_blur.kernel_size=$BLUR_KERNEL_SIZE"
    "eval.pixels_gaussian_blur.sigma_min=$BLUR_SIGMA_MIN"
    "eval.pixels_gaussian_blur.sigma_max=$BLUR_SIGMA_MAX"
  )
  echo "Eval pixel blur enabled: probability=$BLUR_PROBABILITY kernel_size=$BLUR_KERNEL_SIZE sigma=[$BLUR_SIGMA_MIN,$BLUR_SIGMA_MAX]"
else
  echo "Eval pixel blur disabled."
fi

read -r -a TASKS <<< "$TASK_LIST"
read -r -a SEEDS <<< "$SEED_LIST"

echo "--- PLANNING CONFIG ---"
echo "EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "MODEL_RUN=$MODEL_RUN"
echo "DATASET_NAME=$DATASET_NAME"
echo "TASK_LIST=$TASK_LIST"
echo "SEED_LIST=$SEED_LIST"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "EVAL_BUDGET=$EVAL_BUDGET"
echo "HORIZON=$HORIZON"
echo "RECEDING_HORIZON=$RECEDING_HORIZON"
echo "WARM_START=$WARM_START"
echo "CLAMP_ACTION_CANDIDATES=$CLAMP_ACTION_CANDIDATES"
echo "CACHE_ALL_LOADED=$CACHE_ALL_LOADED"
echo "SAVE_VIDEO=$SAVE_VIDEO"
echo "-----------------------"

for task in "${TASKS[@]}"; do
  task_slug="${task//-/_}"
  result_file="$OUTPUT_DIR/${task_slug}.txt"
  for seed in "${SEEDS[@]}"; do
    CMD=(
      srun python -u eval.py
      --config-name=metaworld
      "policy=$MODEL_RUN"
      "eval.dataset_name=$DATASET_NAME"
      "world.metaworld_env_name=$task"
      "world.history_size=$WORLD_HISTORY_SIZE"
      "seed=$seed"
      "eval.num_eval=$EVAL_NUM"
      "eval.goal_offset_steps=$GOAL_OFFSET_STEPS"
      "eval.eval_budget=$EVAL_BUDGET"
      "dataset.cache_all_loaded=$CACHE_ALL_LOADED"
      "plan_config.horizon=$HORIZON"
      "plan_config.receding_horizon=$RECEDING_HORIZON"
      "plan_config.action_block=$ACTION_BLOCK"
      "+plan_config.warm_start=$WARM_START"
      "solver.var_scale=$SOLVER_VAR_SCALE"
      "solver.num_samples=$CEM_NUM_SAMPLES"
      "solver.topk=$CEM_TOPK"
      "solver.n_steps=$CEM_STEPS"
      "eval.clamp_action_candidates=$CLAMP_ACTION_CANDIDATES"
      "eval.first_action_delta_limit=$FIRST_ACTION_DELTA_LIMIT"
      "eval.action_delta_limit=$ACTION_DELTA_LIMIT"
      "eval.execution_action_delta_limit=$EXECUTION_ACTION_DELTA_LIMIT"
      "eval.execution_action_norm_limit=$EXECUTION_ACTION_NORM_LIMIT"
      "eval.action_cost.norm_weight=$ACTION_NORM_WEIGHT"
      "eval.action_cost.delta_weight=$ACTION_DELTA_WEIGHT"
      "eval.action_cost.first_delta_weight=$FIRST_ACTION_DELTA_WEIGHT"
      "+eval.save_video=$SAVE_VIDEO"
      "${DROP_MODALITIES_OVERRIDE[@]}"
      "${BLUR_OVERRIDES[@]}"
      "output.filename=$result_file"
    )
    printf 'Running command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    "${CMD[@]}"
  done
done
