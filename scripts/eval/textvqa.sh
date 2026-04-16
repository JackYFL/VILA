#!/bin/bash
# Evaluate on TextVQA validation set (5000 samples).
# Usage: bash scripts/eval/textvqa.sh <model_path> <conv_mode> [max_tiles]
# Example: bash scripts/eval/textvqa.sh Efficient-Large-Model/NVILA-8B hermes-2 12
#
# Requires VILA_DATASETS=localssd (or a registry that defines "textvqa").
# The localssd registry points to /mnt/localssd/data/eval/textvqa/.

set -e

MODEL_PATH=$1
CONV_MODE=$2
MAX_TILES=${3:-12}

MODEL_NAME=$(basename $MODEL_PATH)
OUTPUT_DIR=${OUTPUT_DIR:-"runs/eval/$MODEL_NAME/textvqa"}

NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
GENERATION_CONFIG='{"max_new_tokens": 16}'

export VILA_DATASETS=${VILA_DATASETS:-"localssd"}

torchrun --nproc-per-node=$NPROC_PER_NODE \
    llava/eval/textvqa.py \
    --model-path $MODEL_PATH \
    --conv-mode $CONV_MODE \
    --max-tiles $MAX_TILES \
    --generation-config "$GENERATION_CONFIG" \
    --output-dir $OUTPUT_DIR
