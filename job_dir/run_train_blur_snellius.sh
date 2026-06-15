#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=TRAIN_BLUR
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --output=out_job_dir/TRAIN_BLUR_%A.out

set -euo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1

export STABLEWM_HOME="${STABLEWM_HOME:-/home/dpasero/project_space}"

cd MultiSensorRoboticsWM/
source .venv/bin/activate

DATASET_NAME="${DATASET_NAME:-metaworld_button_press}"
OUTPUT_MODEL_NAME="${OUTPUT_MODEL_NAME:-metaworld_selfattention_button_press_5_blur}"
SUBDIR="${SUBDIR:-$OUTPUT_MODEL_NAME}"
BATCH_SIZE="${BATCH_SIZE:-600}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"

set +e
srun python train.py \
  data=metaworld \
  obs_encoder=multimodal \
  obs_encoder.imputer.type=missing_token \
  obs_encoder.imputer.random_mask_prob=0.3 \
  obs_encoder.fusion.type=selfattention \
  data.dataset.name="$DATASET_NAME" \
  data.dataset.keys_to_cache='[action,proprio,force_torque,depth,tactile,pixels]' \
  num_workers=1 \
  loader.prefetch_factor=1 \
  loader.pin_memory=True \
  output_model_name="$OUTPUT_MODEL_NAME" \
  subdir="$SUBDIR" \
  trainer.max_epochs="$MAX_EPOCHS" \
  loader.batch_size="$BATCH_SIZE" \
  loss.sigreg.weight=0.01 \
  obs_encoder.modalities.pixels.gaussian_blur.enabled=True \
  obs_encoder.modalities.pixels.gaussian_blur.training_only=True \
  obs_encoder.modalities.pixels.gaussian_blur.probability=0.75 \
  obs_encoder.modalities.pixels.gaussian_blur.kernel_size=5 \
  obs_encoder.modalities.pixels.gaussian_blur.sigma_min=0.0 \
  obs_encoder.modalities.pixels.gaussian_blur.sigma_max=1.5
TRAIN_EXIT_CODE=$?
set -e

exit "${TRAIN_EXIT_CODE}"
