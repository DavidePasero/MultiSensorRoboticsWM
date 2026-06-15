#!/usr/bin/env bash
set -euo pipefail

# Run this from the Snellius login node:
#   bash job_dir/run_planning_suite_snellius.sh

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p out_job_dir

RUNNER="${RUNNER:-job_dir/run_planning_snellius.sh}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/dpasero/project_space/documentation/planning_corrected_eval70_${RUN_STAMP}}"
SEED_LIST="${SEED_LIST:-42 43 44}"

COMMON_EVAL_BUDGET="${EVAL_BUDGET:-70}"
COMMON_GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-20}"
COMMON_HORIZON="${HORIZON:-15}"
COMMON_RECEDING_HORIZON="${RECEDING_HORIZON:-5}"
COMMON_WARM_START="${WARM_START:-true}"
COMMON_CLAMP_ACTION_CANDIDATES="${CLAMP_ACTION_CANDIDATES:-false}"
COMMON_CACHE_ALL_LOADED="${CACHE_ALL_LOADED:-true}"
COMMON_SAVE_VIDEO="${SAVE_VIDEO:-true}"

submit_planning() {
  local experiment_name="$1"
  local model_run="$2"
  local dataset_name="$3"
  local task_name="$4"
  local keep_modalities_csv="${5:-all}"
  local modality_substitution="${6:-impute}"
  local blur="${7:-false}"
  local blur_kernel_size="${8:-5}"
  local blur_sigma="${9:-1.0}"

  local job_name="PLAN_${experiment_name//-/_}"
  job_name="${job_name//\//_}"
  job_name="${job_name:0:60}"

  (
    export EXPERIMENT_NAME="$experiment_name"
    export OUTPUT_ROOT="$OUTPUT_ROOT"
    export MODEL_RUN="$model_run"
    export DATASET_NAME="$dataset_name"
    export TASK_LIST="$task_name"
    export SEED_LIST="$SEED_LIST"

    export EVAL_BUDGET="$COMMON_EVAL_BUDGET"
    export GOAL_OFFSET_STEPS="$COMMON_GOAL_OFFSET_STEPS"
    export HORIZON="$COMMON_HORIZON"
    export RECEDING_HORIZON="$COMMON_RECEDING_HORIZON"
    export WARM_START="$COMMON_WARM_START"
    export CLAMP_ACTION_CANDIDATES="$COMMON_CLAMP_ACTION_CANDIDATES"
    export CACHE_ALL_LOADED="$COMMON_CACHE_ALL_LOADED"
    export SAVE_VIDEO="$COMMON_SAVE_VIDEO"

    export KEEP_MODALITIES_CSV="$keep_modalities_csv"
    export MODALITY_SUBSTITUTION="$modality_substitution"

    export BLUR="$blur"
    export BLUR_PROBABILITY="1.0"
    export BLUR_KERNEL_SIZE="$blur_kernel_size"
    export BLUR_SIGMA_MIN="$blur_sigma"
    export BLUR_SIGMA_MAX="$blur_sigma"

    # Keep all post-hoc action constraints disabled for the corrected rerun.
    export FIRST_ACTION_DELTA_LIMIT="null"
    export ACTION_DELTA_LIMIT="null"
    export EXECUTION_ACTION_DELTA_LIMIT="null"
    export EXECUTION_ACTION_NORM_LIMIT="null"
    export ACTION_NORM_WEIGHT="0.0"
    export ACTION_DELTA_WEIGHT="0.0"
    export FIRST_ACTION_DELTA_WEIGHT="0.0"

    job_id="$(sbatch --parsable --job-name "$job_name" --export=ALL "$RUNNER")"
    echo "Submitted $job_id  $experiment_name"
  )
}

echo "Submitting corrected planning suite."
echo "Run stamp: $RUN_STAMP"
echo "Output root: $OUTPUT_ROOT"
echo "Seeds: $SEED_LIST"
echo "Eval budget: $COMMON_EVAL_BUDGET"
echo "Warm start: $COMMON_WARM_START"
echo "Action clamping: $COMMON_CLAMP_ACTION_CANDIDATES"
echo ""

# ---------------------------------------------------------------------------
# Normal planning: button-press and drawer-open.
# ---------------------------------------------------------------------------
submit_planning \
  "normal_button_press_pixels" \
  "button_press/metaworld_pixels_button_press_2" \
  "metaworld_eval_button_press" \
  "button-press-v3"

submit_planning \
  "normal_button_press_selfattention" \
  "button_press/metaworld_selfattention_button_press_low_sigreg" \
  "metaworld_eval_button_press" \
  "button-press-v3"

submit_planning \
  "normal_button_press_gated" \
  "button_press/metaworld_gated_button_press" \
  "metaworld_eval_button_press" \
  "button-press-v3"

submit_planning \
  "normal_button_press_coproj" \
  "button_press/metaworld_coproj_button_press" \
  "metaworld_eval_button_press" \
  "button-press-v3"

submit_planning \
  "normal_button_press_random" \
  "random" \
  "metaworld_eval_button_press" \
  "button-press-v3"

submit_planning \
  "normal_drawer_open_pixels" \
  "drawer_open/metaworld_pixels_drawer_open" \
  "metaworld_eval_drawer_open" \
  "drawer-open-v3"

submit_planning \
  "normal_drawer_open_selfattention" \
  "drawer_open/metaworld_selfattention_drawer_open_low_sigreg" \
  "metaworld_eval_drawer_open" \
  "drawer-open-v3"

submit_planning \
  "normal_drawer_open_gated" \
  "drawer_open/metaworld_gated_drawer_open" \
  "metaworld_eval_drawer_open" \
  "drawer-open-v3"

submit_planning \
  "normal_drawer_open_coproj" \
  "drawer_open/metaworld_coproj_drawer_open" \
  "metaworld_eval_drawer_open" \
  "drawer-open-v3"

submit_planning \
  "normal_drawer_open_random" \
  "random" \
  "metaworld_eval_drawer_open" \
  "drawer-open-v3"

# ---------------------------------------------------------------------------
# Missing-modality planning: keep only pixels on button-press.
# Pixels baseline receives no modality drop because it only uses pixels anyway.
# ---------------------------------------------------------------------------
submit_planning \
  "missing_button_press_only_pixels_selfmask" \
  "button_press/metaworld_selfattention_selfmask_button_press" \
  "metaworld_eval_button_press" \
  "button-press-v3" \
  "pixels" \
  "impute"

submit_planning \
  "missing_button_press_only_pixels_missing_token" \
  "button_press/metaworld_selfattention_masked_button_press" \
  "metaworld_eval_button_press" \
  "button-press-v3" \
  "pixels" \
  "impute"

submit_planning \
  "missing_button_press_only_pixels_latent_reconstruction" \
  "button_press/metaworld_selfattention_latent_reconstruction_button_press" \
  "metaworld_eval_button_press" \
  "button-press-v3" \
  "pixels" \
  "impute"

submit_planning \
  "missing_button_press_only_pixels_pixels_baseline" \
  "button_press/metaworld_pixels_button_press_2" \
  "metaworld_eval_button_press" \
  "button-press-v3" \
  "all" \
  "impute"

# ---------------------------------------------------------------------------
# Blur planning: button-press, depth masked, three increasing blur strengths.
# ---------------------------------------------------------------------------
BLUR_KEEP_MODALITIES="pixels,tactile,proprio,force_torque"

for blur_level in \
  "k05_sigma1 5 1.0" \
  "k11_sigma2 11 2.0" \
  "k19_sigma3 19 3.0"
do
  read -r blur_name blur_kernel blur_sigma <<< "$blur_level"

  submit_planning \
    "blur_button_press_${blur_name}_latent_reconstruction" \
    "blur/metaworld_selfattention_latent_reconstruction_blur_button_press" \
    "metaworld_eval_button_press" \
    "button-press-v3" \
    "$BLUR_KEEP_MODALITIES" \
    "impute" \
    "true" \
    "$blur_kernel" \
    "$blur_sigma"

  submit_planning \
    "blur_button_press_${blur_name}_missing_token" \
    "blur/metaworld_selfattention_masked_button_press_5_blur" \
    "metaworld_eval_button_press" \
    "button-press-v3" \
    "$BLUR_KEEP_MODALITIES" \
    "impute" \
    "true" \
    "$blur_kernel" \
    "$blur_sigma"

  submit_planning \
    "blur_button_press_${blur_name}_selfmask" \
    "blur/metaworld_selfattention_selfmask_blur_button_press" \
    "metaworld_eval_button_press" \
    "button-press-v3" \
    "$BLUR_KEEP_MODALITIES" \
    "impute" \
    "true" \
    "$blur_kernel" \
    "$blur_sigma"
done

echo ""
echo "Submitted all planning jobs."
