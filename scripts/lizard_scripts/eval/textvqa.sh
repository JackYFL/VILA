#!/bin/bash
# Evaluate a Lizard stage2 checkpoint on TextVQA validation set.
#
# Stage2 checkpoints (LoRA + LizardAttention on NVILA-8B) require a custom
# 4-step loading procedure that plain llava.load() does not support.
# This script uses llava/eval/textvqa_lizard.py which handles that loading.
#
# Usage:
#   bash scripts/lizard_scripts/eval/textvqa.sh [max_tiles]
#
# Example:
#   bash scripts/lizard_scripts/eval/textvqa.sh 12
#
# Override defaults via env vars:
#   BASE_MODEL=...   STAGE1_CKPT=...   STAGE2_CKPT=...   OUTPUT_DIR=...

set -e

MAX_TILES=${1:-12}

# ── Checkpoint paths ──────────────────────────────────────────────────────────
BASE_MODEL=${BASE_MODEL:-"Efficient-Large-Model/NVILA-8B"}

STAGE1_CKPT=${STAGE1_CKPT:-"runs/train/nvila-8b-llava-onevision-img-stage1/model/checkpoint-1500"}

STAGE2_CKPT=${STAGE2_CKPT:-"runs/train/nvila-8b-llava-onevision-img-stage2-new/model/checkpoint-20194"}

# Conv mode used during training (Qwen2 ChatML)
CONV_MODE=${CONV_MODE:-"hermes-2"}

# ── Output ────────────────────────────────────────────────────────────────────
STAGE2_NAME=$(basename $(dirname $STAGE2_CKPT))/$(basename $STAGE2_CKPT)
USER_SET_OUTPUT_DIR=${OUTPUT_DIR+x}      # set iff user exported OUTPUT_DIR
OUTPUT_DIR_OVERRIDE=${OUTPUT_DIR:-""}   # capture whether user set it
OUTPUT_DIR=${OUTPUT_DIR:-"runs/eval/lizard/${STAGE2_NAME}/textvqa"}

# ── Runtime ───────────────────────────────────────────────────────────────────
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
GENERATION_CONFIG='{"max_new_tokens": 16}'

# Cache mode (default: no-cache/full-sequence to match training behavior)
# Set USE_CACHE=1 to enable recurrent cache for speed (may hurt quality).
USE_CACHE=${USE_CACHE:-0}

# Gate diagnostic: set LIZARD_FORCE_GATE=0.99 to override g to a fixed value,
# diagnosing whether gate collapse (g≈0.5 → 2-token memory) is the root cause.
# Default: 0 (use trained gate values).
# Example: LIZARD_FORCE_GATE=0.99 bash scripts/lizard_scripts/eval/textvqa.sh
export LIZARD_FORCE_GATE=${LIZARD_FORCE_GATE:-0}

export VILA_DATASETS=${VILA_DATASETS:-"localssd"}

CACHE_FLAG="--use-cache"
if [ "$USE_CACHE" = "0" ]; then
    CACHE_FLAG="--no-cache"
fi

# If forcing gate (and user hasn't set OUTPUT_DIR explicitly), use a separate dir
if [ "$LIZARD_FORCE_GATE" != "0" ] && [ -n "$LIZARD_FORCE_GATE" ] && [ -z "$USER_SET_OUTPUT_DIR" ]; then
    OUTPUT_DIR="runs/eval/lizard/${STAGE2_NAME}/textvqa_gate${LIZARD_FORCE_GATE}"
fi

echo "BASE_MODEL      = ${BASE_MODEL}"
echo "STAGE1_CKPT     = ${STAGE1_CKPT}"
echo "STAGE2_CKPT     = ${STAGE2_CKPT}"
echo "OUTPUT_DIR      = ${OUTPUT_DIR}"
echo "MAX_TILES       = ${MAX_TILES}"
echo "NPROC           = ${NPROC_PER_NODE}"
echo "CACHE_FLAG      = ${CACHE_FLAG}"
echo "LIZARD_FORCE_GATE = ${LIZARD_FORCE_GATE}"

torchrun --nproc-per-node=$NPROC_PER_NODE \
    llava/eval/textvqa_lizard.py \
    --base-model   "$BASE_MODEL"   \
    --stage1-ckpt  "$STAGE1_CKPT" \
    --stage2-ckpt  "$STAGE2_CKPT" \
    --conv-mode    "$CONV_MODE"   \
    --max-tiles    "$MAX_TILES"   \
    --generation-config "$GENERATION_CONFIG" \
    --output-dir   "$OUTPUT_DIR"  \
    $CACHE_FLAG
