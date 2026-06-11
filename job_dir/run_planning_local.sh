#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

MODEL_RUN="${MODEL_RUN:-button_press/metaworld_selfattention_button_press_5}"
DATASET_NAME="${DATASET_NAME:-metaworld_eval_button_press}"
MODEL_SLUG="${MODEL_RUN//\//__}"
OUTPUT_FILENAME="${OUTPUT_FILENAME:-planning_results_${MODEL_SLUG}.txt}"
WORLD_HISTORY_SIZE="${WORLD_HISTORY_SIZE:-3}"
EVAL_NUM="${EVAL_NUM:-10}"
GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-20}"
EVAL_BUDGET="${EVAL_BUDGET:-50}"
HORIZON="${HORIZON:-20}"
RECEDING_HORIZON="${RECEDING_HORIZON:-5}"
ACTION_BLOCK="${ACTION_BLOCK:-1}"
SOLVER_VAR_SCALE="${SOLVER_VAR_SCALE:-0.3}"
CEM_NUM_SAMPLES="${CEM_NUM_SAMPLES:-300}"
CEM_TOPK="${CEM_TOPK:-30}"
CEM_STEPS="${CEM_STEPS:-30}"
SAVE_VIDEO="${SAVE_VIDEO:-true}"

TASK_LIST="${TASK_LIST:-button-press-v3}"
SEED_LIST="${SEED_LIST:-42 43 44}"
read -r -a TASKS <<< "$TASK_LIST"
read -r -a SEEDS <<< "$SEED_LIST"

BLUR="${BLUR:-false}"
BLUR_PROBABILITY="${BLUR_PROBABILITY:-1.0}"
BLUR_KERNEL_SIZE="${BLUR_KERNEL_SIZE:-19}"
BLUR_SIGMA_MIN="${BLUR_SIGMA_MIN:-3.0}"
BLUR_SIGMA_MAX="${BLUR_SIGMA_MAX:-3.0}"

ALL_MODALITIES=(
  "pixels"
  "depth"
  "tactile"
  "proprio"
  "force_torque"
)

# Leave empty to keep every modality. Otherwise, list the modalities the model
# should receive; every other modality is dropped so the imputer has to fill it.
KEEP_MODALITIES=(

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

DROP_MODALITIES=()
if [[ "${#KEEP_MODALITIES[@]}" -gt 0 ]]; then
  for modality in "${ALL_MODALITIES[@]}"; do
    if ! contains_modality "$modality" "${KEEP_MODALITIES[@]}"; then
      DROP_MODALITIES+=("$modality")
    fi
  done
fi

DROP_MODALITIES_OVERRIDE=()
if [[ "${#DROP_MODALITIES[@]}" -gt 0 ]]; then
  DROP_MODALITIES_OVERRIDE=("eval.drop_modalities=$(join_hydra_list "${DROP_MODALITIES[@]}")")
  echo "Keeping modalities: $(join_hydra_list "${KEEP_MODALITIES[@]}")"
  echo "Dropping modalities: $(join_hydra_list "${DROP_MODALITIES[@]}")"
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

for task in "${TASKS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    CMD=(
      python eval.py
      --config-name=metaworld \
      "policy=$MODEL_RUN" \
      "eval.dataset_name=$DATASET_NAME" \
      "world.metaworld_env_name=$task" \
      "world.history_size=$WORLD_HISTORY_SIZE" \
      "seed=$seed" \
      "eval.num_eval=$EVAL_NUM" \
      "eval.goal_offset_steps=$GOAL_OFFSET_STEPS" \
      "eval.eval_budget=$EVAL_BUDGET" \
      "plan_config.horizon=$HORIZON" \
      "plan_config.receding_horizon=$RECEDING_HORIZON" \
      "plan_config.action_block=$ACTION_BLOCK" \
      "+plan_config.warm_start=false" \
      "solver.var_scale=$SOLVER_VAR_SCALE" \
      "solver.num_samples=$CEM_NUM_SAMPLES" \
      "solver.topk=$CEM_TOPK" \
      "solver.n_steps=$CEM_STEPS" \
      "+eval.save_video=$SAVE_VIDEO" \
      "${DROP_MODALITIES_OVERRIDE[@]}" \
      "${BLUR_OVERRIDES[@]}" \
      "output.filename=$OUTPUT_FILENAME"
    )
    printf 'Running command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    "${CMD[@]}"
  done
done
