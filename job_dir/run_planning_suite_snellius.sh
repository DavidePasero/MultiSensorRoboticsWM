#!/usr/bin/env bash
set -euo pipefail

# Run this from the Snellius login node:
#   bash /home/dpasero/project_space/MultiSensorRoboticsWM/job_dir/run_planning_suite_snellius.sh

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)}"
cd "$PROJECT_DIR"

RUNNER="${RUNNER:-$PROJECT_DIR/job_dir/run_planning_snellius.sh}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/dpasero/project_space/documentation/planning_corrected_eval70_${RUN_STAMP}}"
MANIFEST="${MANIFEST:-$PROJECT_DIR/out_job_dir/planning_suite_${RUN_STAMP}.tsv}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-$PROJECT_DIR/out_job_dir}"
SUBMIT_SUMMARY="${SUBMIT_SUMMARY:-$PROJECT_DIR/out_job_dir/planning_suite_${RUN_STAMP}_submission.txt}"
MAX_ARRAY_PARALLEL="${MAX_ARRAY_PARALLEL:-8}"
SEED_LIST="${SEED_LIST:-42 43 44}"

COMMON_EVAL_NUM="${EVAL_NUM:-10}"
COMMON_EVAL_BUDGET="${EVAL_BUDGET:-70}"
COMMON_GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-20}"
COMMON_HORIZON="${HORIZON:-15}"
COMMON_RECEDING_HORIZON="${RECEDING_HORIZON:-5}"
COMMON_WARM_START="${WARM_START:-true}"
COMMON_CLAMP_ACTION_CANDIDATES="${CLAMP_ACTION_CANDIDATES:-false}"
COMMON_CACHE_ALL_LOADED="${CACHE_ALL_LOADED:-true}"
COMMON_SAVE_VIDEO="${SAVE_VIDEO:-true}"

mkdir -p "$(dirname "$MANIFEST")" "$SLURM_LOG_DIR"
: > "$MANIFEST"

add_planning() {
  local experiment_name="$1"
  local model_run="$2"
  local dataset_name="$3"
  local task_name="$4"
  local keep_modalities_csv="${5:-all}"
  local modality_substitution="${6:-impute}"
  local blur="${7:-false}"
  local blur_kernel_size="${8:-5}"
  local blur_sigma="${9:-1.0}"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$experiment_name" \
    "$model_run" \
    "$dataset_name" \
    "$task_name" \
    "$keep_modalities_csv" \
    "$modality_substitution" \
    "$blur" \
    "$blur_kernel_size" \
    "$blur_sigma" \
    >> "$MANIFEST"
}

echo "Submitting corrected planning suite."
echo "Project dir: $PROJECT_DIR"
echo "Runner: $RUNNER"
echo "Run stamp: $RUN_STAMP"
echo "Output root: $OUTPUT_ROOT"
echo "Manifest: $MANIFEST"
echo "Slurm log dir: $SLURM_LOG_DIR"
echo "Seeds: $SEED_LIST"
echo "Eval episodes per seed: $COMMON_EVAL_NUM"
echo "Eval budget: $COMMON_EVAL_BUDGET"
echo "Warm start: $COMMON_WARM_START"
echo "Action clamping: $COMMON_CLAMP_ACTION_CANDIDATES"
echo "Max concurrent array tasks: $MAX_ARRAY_PARALLEL"
echo ""

# ---------------------------------------------------------------------------
# Normal planning: button-press and drawer-open.
# ---------------------------------------------------------------------------
add_planning \
  "normal_button_press_pixels" \
  "button_press/metaworld_pixels_button_press_2" \
  "metaworld_eval_button_press" \
  "button-press-v3"

add_planning \
  "normal_button_press_selfattention" \
  "button_press/metaworld_selfattention_button_press_low_sigreg" \
  "metaworld_eval_button_press" \
  "button-press-v3"

add_planning \
  "normal_button_press_gated" \
  "button_press/metaworld_gated_button_press" \
  "metaworld_eval_button_press" \
  "button-press-v3"

add_planning \
  "normal_button_press_coproj" \
  "button_press/metaworld_coproj_button_press" \
  "metaworld_eval_button_press" \
  "button-press-v3"

add_planning \
  "normal_button_press_random" \
  "random" \
  "metaworld_eval_button_press" \
  "button-press-v3"

add_planning \
  "normal_drawer_open_pixels" \
  "drawer_open/metaworld_pixels_drawer_open" \
  "metaworld_eval_drawer_open" \
  "drawer-open-v3"

add_planning \
  "normal_drawer_open_selfattention" \
  "drawer_open/metaworld_selfattention_drawer_open_low_sigreg" \
  "metaworld_eval_drawer_open" \
  "drawer-open-v3"

add_planning \
  "normal_drawer_open_gated" \
  "drawer_open/metaworld_gated_drawer_open" \
  "metaworld_eval_drawer_open" \
  "drawer-open-v3"

add_planning \
  "normal_drawer_open_coproj" \
  "drawer_open/metaworld_coproj_drawer_open" \
  "metaworld_eval_drawer_open" \
  "drawer-open-v3"

add_planning \
  "normal_drawer_open_random" \
  "random" \
  "metaworld_eval_drawer_open" \
  "drawer-open-v3"

# ---------------------------------------------------------------------------
# Missing-modality planning: keep only pixels on button-press.
# Pixels baseline receives no modality drop because it only uses pixels anyway.
# ---------------------------------------------------------------------------
add_planning \
  "missing_button_press_only_pixels_selfmask" \
  "button_press/metaworld_selfattention_selfmask_button_press" \
  "metaworld_eval_button_press" \
  "button-press-v3" \
  "pixels" \
  "impute"

add_planning \
  "missing_button_press_only_pixels_missing_token" \
  "button_press/metaworld_selfattention_masked_button_press" \
  "metaworld_eval_button_press" \
  "button-press-v3" \
  "pixels" \
  "impute"

add_planning \
  "missing_button_press_only_pixels_latent_reconstruction" \
  "button_press/metaworld_selfattention_latent_reconstruction_button_press" \
  "metaworld_eval_button_press" \
  "button-press-v3" \
  "pixels" \
  "impute"

add_planning \
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

  add_planning \
    "blur_button_press_${blur_name}_latent_reconstruction" \
    "blur/metaworld_selfattention_latent_reconstruction_blur_button_press" \
    "metaworld_eval_button_press" \
    "button-press-v3" \
    "$BLUR_KEEP_MODALITIES" \
    "impute" \
    "true" \
    "$blur_kernel" \
    "$blur_sigma"

  add_planning \
    "blur_button_press_${blur_name}_missing_token" \
    "blur/metaworld_selfattention_masked_button_press_5_blur" \
    "metaworld_eval_button_press" \
    "button-press-v3" \
    "$BLUR_KEEP_MODALITIES" \
    "impute" \
    "true" \
    "$blur_kernel" \
    "$blur_sigma"

  add_planning \
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
NUM_JOBS="$(wc -l < "$MANIFEST")"
NUM_JOBS="${NUM_JOBS//[[:space:]]/}"
if [[ "$NUM_JOBS" -lt 1 ]]; then
  echo "No planning jobs were written to $MANIFEST." >&2
  exit 1
fi

ARRAY_SPEC="1-${NUM_JOBS}"
if [[ -n "$MAX_ARRAY_PARALLEL" && "$MAX_ARRAY_PARALLEL" != "0" ]]; then
  ARRAY_SPEC="${ARRAY_SPEC}%${MAX_ARRAY_PARALLEL}"
fi

export PLANNING_MANIFEST="$MANIFEST"
export OUTPUT_ROOT
export SEED_LIST
export EVAL_NUM="$COMMON_EVAL_NUM"
export EVAL_BUDGET="$COMMON_EVAL_BUDGET"
export GOAL_OFFSET_STEPS="$COMMON_GOAL_OFFSET_STEPS"
export HORIZON="$COMMON_HORIZON"
export RECEDING_HORIZON="$COMMON_RECEDING_HORIZON"
export WARM_START="$COMMON_WARM_START"
export CLAMP_ACTION_CANDIDATES="$COMMON_CLAMP_ACTION_CANDIDATES"
export CACHE_ALL_LOADED="$COMMON_CACHE_ALL_LOADED"
export SAVE_VIDEO="$COMMON_SAVE_VIDEO"
export BLUR_PROBABILITY="1.0"

# Keep all post-hoc action constraints disabled for the corrected rerun.
export FIRST_ACTION_DELTA_LIMIT="null"
export ACTION_DELTA_LIMIT="null"
export EXECUTION_ACTION_DELTA_LIMIT="null"
export EXECUTION_ACTION_NORM_LIMIT="null"
export ACTION_NORM_WEIGHT="0.0"
export ACTION_DELTA_WEIGHT="0.0"
export FIRST_ACTION_DELTA_WEIGHT="0.0"

JOB_NAME="PLAN_SUITE_${RUN_STAMP}"
JOB_NAME="${JOB_NAME:0:60}"
JOB_ID="$(
  sbatch \
    --parsable \
    --array="$ARRAY_SPEC" \
    --job-name "$JOB_NAME" \
    --chdir="$PROJECT_DIR" \
    --output="$SLURM_LOG_DIR/${JOB_NAME}_%A_%a.out" \
    --error="$SLURM_LOG_DIR/${JOB_NAME}_%A_%a.out" \
    --export=ALL \
    "$RUNNER"
)"

echo "Wrote $NUM_JOBS planning configurations to $MANIFEST."
echo "Submitted array job $JOB_ID with --array=$ARRAY_SPEC."

cat > "$SUBMIT_SUMMARY" <<EOF
Planning suite submission
=========================
job_id: $JOB_ID
array_spec: $ARRAY_SPEC
run_stamp: $RUN_STAMP
output_root: $OUTPUT_ROOT
manifest: $MANIFEST
slurm_log_pattern: $SLURM_LOG_DIR/${JOB_NAME}_${JOB_ID}_<array_task_id>.out
seeds: $SEED_LIST
eval_num_per_seed: $COMMON_EVAL_NUM
eval_budget: $COMMON_EVAL_BUDGET
warm_start: $COMMON_WARM_START
action_clamping: $COMMON_CLAMP_ACTION_CANDIDATES
save_video: $COMMON_SAVE_VIDEO

Useful commands:
  sacct -j $JOB_ID --format=JobID,JobName%35,State,ExitCode,Elapsed -X
  ls -ltr $SLURM_LOG_DIR/${JOB_NAME}_${JOB_ID}_*.out
  nl -ba $MANIFEST
  find $OUTPUT_ROOT -maxdepth 3 -type f | sort
EOF

echo "Submission summary: $SUBMIT_SUMMARY"
