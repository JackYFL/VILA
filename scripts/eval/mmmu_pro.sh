#!/bin/bash
# Evaluate on MMMU-Pro (all three configs by default).
# Usage: bash scripts/eval/mmmu_pro.sh <model_path> <conv_mode> [max_tiles]
# Example: bash scripts/eval/mmmu_pro.sh Efficient-Large-Model/NVILA-8B hermes-2 12
#
# To run a single config:
#   PRO_CONFIGS="standard (4 options)" bash scripts/eval/mmmu_pro.sh ...
#   PRO_CONFIGS="vision" bash scripts/eval/mmmu_pro.sh ...

set -e

MODEL_PATH=$1
CONV_MODE=$2
MAX_TILES=${3:-12}

MODEL_NAME=$(basename $MODEL_PATH)
OUTPUT_DIR=${OUTPUT_DIR:-"runs/eval/$MODEL_NAME/mmmu_pro"}

NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
GENERATION_CONFIG='{"max_new_tokens": 16}'
PRO_DATA_PATH=${MMMU_PRO_DATA_PATH:-"/mnt/localssd/data/eval/mmmu_pro"}

# Build optional --pro-configs argument
PRO_CONFIGS_ARGS=""
if [ -n "$PRO_CONFIGS" ]; then
    PRO_CONFIGS_ARGS="--pro-configs $PRO_CONFIGS"
fi

torchrun --nproc-per-node=$NPROC_PER_NODE \
    llava/eval/mmmu.py \
    --model-path $MODEL_PATH \
    --conv-mode $CONV_MODE \
    --max-tiles $MAX_TILES \
    --generation-config "$GENERATION_CONFIG" \
    --pro \
    --pro-data-path $PRO_DATA_PATH \
    $PRO_CONFIGS_ARGS \
    --output-dir $OUTPUT_DIR
