#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=TRAIN
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=5:00:00
#SBATCH --output=out_job_dir/TRAIN_%A.out

set -euo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1

export STABLEWM_HOME="/home/dpasero/project_space"

cd MultiSensorRoboticsWM/
source .venv/bin/activate

MODEL_RUN="${MODEL_RUN:-metaworld_selfattention}"
DATASET_NAME="${DATASET_NAME:-metaworld_eval}"
OUTPUT_FILENAME="${OUTPUT_FILENAME:-planning_snellius_results.txt}"

TASKS=(
  "reach-v3"
  "push-v3"
  "pick-place-v3"
  "drawer-open-v3"
  "drawer-close-v3"
  "door-open-v3"
  "button-press-v3"
  "hammer-v3"
)

SEEDS=(42 43 44)

for task in "${TASKS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    eval_num=10
    goal_offset_steps=25
    eval_budget=50
    horizon=25
    receding_horizon=5
    action_block=1

    srun python eval.py \
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
      "output.filename=$OUTPUT_FILENAME"
  done
done
