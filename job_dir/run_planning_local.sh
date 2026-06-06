#!/usr/bin/env bash

source .venv/bin/activate

MODEL_RUN="${MODEL_RUN:-bin_picking/metaworld_selfattention_latent_reconstruction_bin_picking}"
DATASET_NAME="${DATASET_NAME:-metaworld_eval_bin_picking}"
OUTPUT_FILENAME="${OUTPUT_FILENAME:-documentation/planning_results_bin_picking/planning_results_${MODEL_RUN}.txt}"

TASKS=(
  "bin-picking-v3" \
)

SEEDS=(42 43 44)

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
      "output.filename=$OUTPUT_FILENAME"
  done
done
