#!/bin/bash
# Evaluate a Lizard stage2 checkpoint on MMMU validation set (30 subjects, ~900 samples).
#
# Stage2 checkpoints (LoRA + LizardAttention on NVILA-8B) require a custom
# 4-step loading procedure that plain llava.load() does not support.
# This script uses llava/eval/mmmu_lizard.py which handles that loading.
#
# Usage:
#   bash scripts/lizard_scripts/eval/mmmu.sh [max_tiles]
#
# Example:
#   bash scripts/lizard_scripts/eval/mmmu.sh 12
#
# Override defaults via env vars:
#   BASE_MODEL=...   STAGE1_CKPT=...   STAGE2_CKPT=...   OUTPUT_DIR=...
#   SPLIT=validation DATA_PATH=...     PRO=1             USE_CACHE=1

set -e

MAX_TILES=${1:-12}

# ── Checkpoint paths ──────────────────────────────────────────────────────────
BASE_MODEL=${BASE_MODEL:-"Efficient-Large-Model/NVILA-8B"}

# STAGE1_CKPT is optional.  Leave empty to load stage2 alone by consolidating
# its DeepSpeed ZeRO shards (global_step*/).  Set it to fall back to the legacy
# 6-step loader (needed only if global_step*/ was pruned).
STAGE1_CKPT=${STAGE1_CKPT:-""}

STAGE2_CKPT=${STAGE2_CKPT:-"runs/train/nvila-8b-llava-onevision-img-stage2-new/model/checkpoint-20194"}

# Conv mode used during training (Qwen2 ChatML)
CONV_MODE=${CONV_MODE:-"hermes-2"}

# ── MMMU dataset ──────────────────────────────────────────────────────────────
SPLIT=${SPLIT:-"validation"}
DATA_PATH=${MMMU_DATA_PATH:-"/mnt/localssd/data/eval/mmmu"}
PRO=${PRO:-0}
PRO_DATA_PATH=${PRO_DATA_PATH:-"/mnt/localssd/data/eval/mmmu_pro"}

# ── Output ────────────────────────────────────────────────────────────────────
STAGE2_NAME=$(basename $(dirname $STAGE2_CKPT))/$(basename $STAGE2_CKPT)
USER_SET_OUTPUT_DIR=${OUTPUT_DIR+x}
OUTPUT_DIR_OVERRIDE=${OUTPUT_DIR:-""}
if [ "$PRO" = "1" ]; then
    OUTPUT_DIR=${OUTPUT_DIR:-"runs/eval/lizard/${STAGE2_NAME}/mmmu_pro"}
else
    OUTPUT_DIR=${OUTPUT_DIR:-"runs/eval/lizard/${STAGE2_NAME}/mmmu"}
fi

# ── Runtime ───────────────────────────────────────────────────────────────────
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
GENERATION_CONFIG='{"max_new_tokens": 16}'

# Cache mode (default: no-cache/full-sequence to match training behavior).
USE_CACHE=${USE_CACHE:-0}

# Gate diagnostic: set LIZARD_FORCE_GATE=0.99 to override g to a fixed value.
export LIZARD_FORCE_GATE=${LIZARD_FORCE_GATE:-0}

export VILA_DATASETS=${VILA_DATASETS:-"localssd"}

CACHE_FLAG="--use-cache"
if [ "$USE_CACHE" = "0" ]; then
    CACHE_FLAG="--no-cache"
fi

# If forcing gate (and user hasn't set OUTPUT_DIR explicitly), use a separate dir
if [ "$LIZARD_FORCE_GATE" != "0" ] && [ -n "$LIZARD_FORCE_GATE" ] && [ -z "$USER_SET_OUTPUT_DIR" ]; then
    if [ "$PRO" = "1" ]; then
        OUTPUT_DIR="runs/eval/lizard/${STAGE2_NAME}/mmmu_pro_gate${LIZARD_FORCE_GATE}"
    else
        OUTPUT_DIR="runs/eval/lizard/${STAGE2_NAME}/mmmu_gate${LIZARD_FORCE_GATE}"
    fi
fi

echo "BASE_MODEL      = ${BASE_MODEL}"
echo "STAGE1_CKPT     = ${STAGE1_CKPT}"
echo "STAGE2_CKPT     = ${STAGE2_CKPT}"
echo "OUTPUT_DIR      = ${OUTPUT_DIR}"
echo "MAX_TILES       = ${MAX_TILES}"
echo "NPROC           = ${NPROC_PER_NODE}"
echo "CACHE_FLAG      = ${CACHE_FLAG}"
echo "LIZARD_FORCE_GATE = ${LIZARD_FORCE_GATE}"
echo "PRO             = ${PRO}"
if [ "$PRO" = "1" ]; then
    echo "PRO_DATA_PATH   = ${PRO_DATA_PATH}"
else
    echo "SPLIT           = ${SPLIT}"
    echo "DATA_PATH       = ${DATA_PATH}"
fi

EXTRA_ARGS=""
if [ "$PRO" = "1" ]; then
    EXTRA_ARGS="--pro --pro-data-path ${PRO_DATA_PATH}"
else
    EXTRA_ARGS="--split ${SPLIT} --data-path ${DATA_PATH}"
fi

STAGE1_FLAG=""
if [ -n "$STAGE1_CKPT" ]; then
    STAGE1_FLAG="--stage1-ckpt $STAGE1_CKPT"
fi

torchrun --nproc-per-node=$NPROC_PER_NODE \
    llava/eval/mmmu_lizard.py \
    --base-model   "$BASE_MODEL"   \
    $STAGE1_FLAG \
    --stage2-ckpt  "$STAGE2_CKPT" \
    --conv-mode    "$CONV_MODE"   \
    --max-tiles    "$MAX_TILES"   \
    --generation-config "$GENERATION_CONFIG" \
    --output-dir   "$OUTPUT_DIR"  \
    $EXTRA_ARGS \
    $CACHE_FLAG
