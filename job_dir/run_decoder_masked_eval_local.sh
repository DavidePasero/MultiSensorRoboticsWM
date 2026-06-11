#!/usr/bin/env bash

set -euo pipefail

source .venv/bin/activate

DECODER_DIR="${DECODER_DIR:-$HOME/.stable_worldmodel/decoders}"
CACHE_DIR="${CACHE_DIR:-${STABLEWM_HOME:-$HOME/.stable_worldmodel}}"
DATASET_NAME="${DATASET_NAME:-metaworld_eval_button_press}"
OUTPUT_DIR="${OUTPUT_DIR:-documentation/decoder_masked_eval_button_press}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-1}"
DEVICE="${DEVICE:-}"

# Leave empty to run only all_modalities + drop_each_modality conditions.
# Each entry adds one extra keep-only condition. A single word keeps one
# modality; multiple words in one entry keep that group together.
KEEP_MODALITY_GROUPS=(
  "pixels"
  "depth"
  "tactile"
  "proprio"
  "force_torque"
)

KEEP_MODALITIES_ARGS=()
for keep_group in "${KEEP_MODALITY_GROUPS[@]}"; do
  read -r -a keep_modalities <<< "$keep_group"
  KEEP_MODALITIES_ARGS+=(--keep-modalities "${keep_modalities[@]}")
done

DEVICE_ARGS=()
if [[ -n "$DEVICE" ]]; then
  DEVICE_ARGS=(--device "$DEVICE")
fi

echo "--- DECODER MASKED EVAL CONFIG ---"
echo "DECODER_DIR=$DECODER_DIR"
echo "CACHE_DIR=$CACHE_DIR"
echo "DATASET_NAME=$DATASET_NAME"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "NUM_WORKERS=$NUM_WORKERS"
if [[ "${#KEEP_MODALITY_GROUPS[@]}" -gt 0 ]]; then
  echo "KEEP_MODALITY_GROUPS=[${KEEP_MODALITY_GROUPS[*]}]"
else
  echo "KEEP_MODALITY_GROUPS=[]"
fi
echo "----------------------------------"

python -u experiments/evaluate_decoders_masked.py \
  --decoder-dir "$DECODER_DIR" \
  --cache-dir "$CACHE_DIR" \
  --dataset-name "$DATASET_NAME" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  "${KEEP_MODALITIES_ARGS[@]}" \
  "${DEVICE_ARGS[@]}"
