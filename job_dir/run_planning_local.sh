#!/usr/bin/env bash

source .venv/bin/activate

MODEL_RUN="${MODEL_RUN:-bin_picking/metaworld_coproj_bin_picking}"
DATASET_NAME="${DATASET_NAME:-metaworld_eval_bin_picking}"
OUTPUT_FILENAME="${OUTPUT_FILENAME:-documentation/planning_results_bin_picking_masked/planning_results_${MODEL_RUN}.txt}"

TASKS=(
  "bin-picking-v3" \
)

SEEDS=(42 43 44)

BLUR="${BLUR:-false}"
BLUR_PROBABILITY="${BLUR_PROBABILITY:-1.0}"
BLUR_KERNEL_SIZE="${BLUR_KERNEL_SIZE:-5}"
BLUR_SIGMA_MIN="${BLUR_SIGMA_MIN:-0.5}"
BLUR_SIGMA_MAX="${BLUR_SIGMA_MAX:-2.0}"

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
  "pixels"
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
    eval_num=10
    goal_offset_steps=20
    eval_budget=50
    horizon=25
    receding_horizon=5
    action_block=1

    python eval.py \
      --config-name=metaworld \
      "policy=$MODEL_RUN" \
      "eval.dataset_name=$DATASET_NAME" \
      "world.metaworld_env_name=$task" \
      "seed=$seed" \
      "eval.num_eval=$eval_num" \
      "eval.goal_offset_steps=$goal_offset_steps" \
      "eval.eval_budget=$eval_budget" \
      "plan_config.horizon=$horizon" \
      "plan_config.receding_horizon=$receding_horizon" \
      "plan_config.action_block=$action_block" \
      "${DROP_MODALITIES_OVERRIDE[@]}" \
      "${BLUR_OVERRIDES[@]}" \
      "output.filename=$OUTPUT_FILENAME"
  done
done
