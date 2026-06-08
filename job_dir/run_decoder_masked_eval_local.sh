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
# Otherwise, this adds one extra condition that keeps only these modalities and
# masks all the others through the model imputer.
KEEP_MODALITIES=(
  "pixels"
  "depth"
)

KEEP_MODALITIES_ARGS=()
if [[ "${#KEEP_MODALITIES[@]}" -gt 0 ]]; then
  KEEP_MODALITIES_ARGS=(--keep-modalities "${KEEP_MODALITIES[@]}")
fi

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
if [[ "${#KEEP_MODALITIES[@]}" -gt 0 ]]; then
  echo "KEEP_MODALITIES=[${KEEP_MODALITIES[*]}]"
else
  echo "KEEP_MODALITIES=[]"
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
