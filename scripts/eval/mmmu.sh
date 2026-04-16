#!/bin/bash
# Evaluate on MMMU validation set.
# Usage: bash scripts/eval/mmmu.sh <model_path> <conv_mode> [max_tiles]
# Example: bash scripts/eval/mmmu.sh Efficient-Large-Model/NVILA-8B hermes-2 12

set -e

MODEL_PATH=$1
CONV_MODE=$2
MAX_TILES=${3:-12}

MODEL_NAME=$(basename $MODEL_PATH)
OUTPUT_DIR=${OUTPUT_DIR:-"runs/eval/$MODEL_NAME/mmmu"}

NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
GENERATION_CONFIG='{"max_new_tokens": 16}'
DATA_PATH=${MMMU_DATA_PATH:-"/mnt/localssd/data/eval/mmmu"}

torchrun --nproc-per-node=$NPROC_PER_NODE \
    llava/eval/mmmu.py \
    --model-path $MODEL_PATH \
    --conv-mode $CONV_MODE \
    --max-tiles $MAX_TILES \
    --generation-config "$GENERATION_CONFIG" \
    --split validation \
    --data-path $DATA_PATH \
    --output-dir $OUTPUT_DIR
