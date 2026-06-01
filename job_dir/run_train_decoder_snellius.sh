#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=TRAIN_DECODER
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=18:00:00
#SBATCH --output=out_job_dir/TRAIN_DECODER_%A.out

set -euo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1

export STABLEWM_HOME="${STABLEWM_HOME:-/home/dpasero/project_space}"
export PYTHONUNBUFFERED=1

cd MultiSensorRoboticsWM/
source .venv/bin/activate

echo "--- NODE RESOURCES ---"
echo "Hostname: $(hostname)"
echo "Allocated CPUs according to SLURM: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Visible logical CPUs: $(nproc)"
echo "CPU model:"
lscpu | grep -E 'Model name|Socket|Thread|Core|CPU\(s\)'

echo ""
echo "System RAM:"
free -h

echo ""
echo "SLURM memory allocation:"
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-not set}"
echo "SLURM_MEM_PER_CPU=${SLURM_MEM_PER_CPU:-not set}"
echo "SLURM_JOB_CPUS_PER_NODE=${SLURM_JOB_CPUS_PER_NODE:-not set}"
echo "----------------------"

echo "--- GPU MONITORING START (Interval: 30s) ---"
echo "Timestamp, GPU_Util, Mem_Used, Mem_Total"

gpu_monitor() {
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits
    sleep 30
  done
}

cleanup() {
  if [[ -n "${LOGGER_PID:-}" ]] && kill -0 "${LOGGER_PID}" 2>/dev/null; then
    echo "--- Decoder training finished. Killing GPU logger (PID: ${LOGGER_PID}) ---"
    kill "${LOGGER_PID}" 2>/dev/null || true
    wait "${LOGGER_PID}" 2>/dev/null || true
  fi
}

gpu_monitor &
LOGGER_PID=$!
trap cleanup EXIT INT TERM

CHECKPOINT="${CHECKPOINT:-metaworld_selfattention_drawer_open_high_sigreg/metaworld_selfattention_drawer_open_high_sigreg_epoch_10}"
DECODER_CONFIG="${DECODER_CONFIG:-config/decoder/train_decoder.yaml}"
CACHE_DIR="${CACHE_DIR:-$STABLEWM_HOME}"
DATASET_NAME="${DATASET_NAME:-metaworld_drawer_open}"
TRAIN_ON="${TRAIN_ON:-all}"
MAX_SAMPLES="${MAX_SAMPLES:-50000}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.7}"
VAL_FRACTION="${VAL_FRACTION:-0.15}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
PATIENCE="${PATIENCE:-10}"
SEED="${SEED:-0}"
PIXEL_IMAGE_SIZE="${PIXEL_IMAGE_SIZE:-224}"
PIXEL_PATCH_SIZE="${PIXEL_PATCH_SIZE:-16}"
PIXEL_HIDDEN_DIM="${PIXEL_HIDDEN_DIM:-512}"
PIXEL_NUM_LAYERS="${PIXEL_NUM_LAYERS:-4}"
PIXEL_NUM_HEADS="${PIXEL_NUM_HEADS:-8}"
PIXEL_MLP_RATIO="${PIXEL_MLP_RATIO:-4.0}"
PIXEL_DROPOUT="${PIXEL_DROPOUT:-0.0}"
CONTACT_THRESHOLD="${CONTACT_THRESHOLD:-0.5}"
FORCE_CONTACT_THRESHOLD="${FORCE_CONTACT_THRESHOLD:-1.0}"
TARGETS="${TARGETS:-}"
LOSS_WEIGHTS="${LOSS_WEIGHTS:-}"
CONFIG_PATH="${CONFIG_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

CONFIG_ARGS=()
if [[ -n "$CONFIG_PATH" ]]; then
  CONFIG_ARGS=(--config "$CONFIG_PATH")
fi

TARGET_ARGS=()
if [[ -n "$TARGETS" ]]; then
  read -r -a TARGET_ARRAY <<< "$TARGETS"
  TARGET_ARGS=(--targets "${TARGET_ARRAY[@]}")
fi

LOSS_WEIGHT_ARGS=()
if [[ -n "$LOSS_WEIGHTS" ]]; then
  read -r -a LOSS_WEIGHT_ARRAY <<< "$LOSS_WEIGHTS"
  for loss_weight in "${LOSS_WEIGHT_ARRAY[@]}"; do
    LOSS_WEIGHT_ARGS+=(--loss-weight "$loss_weight")
  done
fi

OUTPUT_ARGS=()
if [[ -n "$OUTPUT_DIR" ]]; then
  OUTPUT_ARGS=(--output-dir "$OUTPUT_DIR")
fi

echo "--- DECODER TRAINING CONFIG ---"
echo "CHECKPOINT=$CHECKPOINT"
echo "DECODER_CONFIG=$DECODER_CONFIG"
echo "CACHE_DIR=$CACHE_DIR"
echo "DATASET_NAME=$DATASET_NAME"
echo "TARGETS=${TARGETS:-default}"
echo "TRAIN_ON=$TRAIN_ON"
echo "MAX_SAMPLES=$MAX_SAMPLES"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "NUM_WORKERS=$NUM_WORKERS"
echo "NUM_EPOCHS=$NUM_EPOCHS"
echo "LR=$LR"
echo "WEIGHT_DECAY=$WEIGHT_DECAY"
echo "OUTPUT_DIR=${OUTPUT_DIR:-auto}"
echo "-------------------------------"

set +e
srun python -u train_decoder.py "$CHECKPOINT" \
  --decoder-config "$DECODER_CONFIG" \
  "${CONFIG_ARGS[@]}" \
  --cache-dir "$CACHE_DIR" \
  --dataset-name "$DATASET_NAME" \
  "${TARGET_ARGS[@]}" \
  "${LOSS_WEIGHT_ARGS[@]}" \
  --train-on "$TRAIN_ON" \
  --max-samples "$MAX_SAMPLES" \
  --train-fraction "$TRAIN_FRACTION" \
  --val-fraction "$VAL_FRACTION" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --num-epochs "$NUM_EPOCHS" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --patience "$PATIENCE" \
  --seed "$SEED" \
  --pixel-image-size "$PIXEL_IMAGE_SIZE" \
  --pixel-patch-size "$PIXEL_PATCH_SIZE" \
  --pixel-hidden-dim "$PIXEL_HIDDEN_DIM" \
  --pixel-num-layers "$PIXEL_NUM_LAYERS" \
  --pixel-num-heads "$PIXEL_NUM_HEADS" \
  --pixel-mlp-ratio "$PIXEL_MLP_RATIO" \
  --pixel-dropout "$PIXEL_DROPOUT" \
  --contact-threshold "$CONTACT_THRESHOLD" \
  --force-contact-threshold "$FORCE_CONTACT_THRESHOLD" \
  "${OUTPUT_ARGS[@]}"
TRAIN_EXIT_CODE=$?
set -e

exit "${TRAIN_EXIT_CODE}"
