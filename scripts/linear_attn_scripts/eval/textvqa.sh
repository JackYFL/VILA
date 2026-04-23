#!/bin/bash
# Evaluate a vanilla-linear-attention stage2 checkpoint on TextVQA validation set.
#
# Stage2 checkpoints (LoRA + VanillaLinearAttention on NVILA-8B) require a
# custom loading procedure that plain llava.load() does not support.
# This script uses llava/eval/textvqa_linear_attn.py which handles that loading.
#
# Usage:
#   bash scripts/linear_attn_scripts/eval/textvqa.sh [max_tiles]
#
# Override defaults via env vars:
#   BASE_MODEL=...   STAGE2_CKPT=...   OUTPUT_DIR=...

set -e

MAX_TILES=${1:-12}

# ── Checkpoint paths ──────────────────────────────────────────────────────────
BASE_MODEL=${BASE_MODEL:-"Efficient-Large-Model/NVILA-8B"}

STAGE2_CKPT=${STAGE2_CKPT:-"runs/train/nvila-8b-llava-onevision-img-linear-attn-stage2/model/checkpoint-20194"}

# Conv mode used during training (Qwen2 ChatML)
CONV_MODE=${CONV_MODE:-"hermes-2"}

# ── Output ────────────────────────────────────────────────────────────────────
STAGE2_NAME=$(basename $(dirname $STAGE2_CKPT))/$(basename $STAGE2_CKPT)
OUTPUT_DIR=${OUTPUT_DIR:-"runs/eval/linear_attn/${STAGE2_NAME}/textvqa"}

# ── Runtime ───────────────────────────────────────────────────────────────────
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
GENERATION_CONFIG='{"max_new_tokens": 16}'

# Cache mode (default: no-cache/full-sequence to match training behavior).
USE_CACHE=${USE_CACHE:-0}

export VILA_DATASETS=${VILA_DATASETS:-"localssd"}

CACHE_FLAG="--use-cache"
if [ "$USE_CACHE" = "0" ]; then
    CACHE_FLAG="--no-cache"
fi

echo "BASE_MODEL      = ${BASE_MODEL}"
echo "STAGE2_CKPT     = ${STAGE2_CKPT}"
echo "OUTPUT_DIR      = ${OUTPUT_DIR}"
echo "MAX_TILES       = ${MAX_TILES}"
echo "NPROC           = ${NPROC_PER_NODE}"
echo "CACHE_FLAG      = ${CACHE_FLAG}"

torchrun --nproc-per-node=$NPROC_PER_NODE \
    llava/eval/textvqa_linear_attn.py \
    --base-model   "$BASE_MODEL"   \
    --stage2-ckpt  "$STAGE2_CKPT" \
    --conv-mode    "$CONV_MODE"   \
    --max-tiles    "$MAX_TILES"   \
    --generation-config "$GENERATION_CONFIG" \
    --output-dir   "$OUTPUT_DIR"  \
    $CACHE_FLAG
