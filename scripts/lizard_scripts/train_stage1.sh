#!/bin/bash
# Train NVILA-8B on LLaVA-OneVision-Data-processed (image SFT stage)
# Run from the VILA root directory: bash scripts/lizard_scripts/train_stage1.sh

set -e

# ============================================================
# Configurable parameters (override via env vars)
# ============================================================
DEFAULT_RUN_NAME="nvila-8b-llava-onevision-img-stage1"
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE=8
DEFAULT_GRADIENT_ACCUMULATION_STEPS=4
DEFAULT_LORA_R=64
DEFAULT_LORA_ALPHA=16
DEFAULT_LORA_DROPOUT=0.05

# Starting checkpoint: pretrained NVILA-8B (stage3 output or HF model)
MODEL_PATH=${MODEL_PATH:-"Efficient-Large-Model/NVILA-8B"}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-"${MODEL_PATH}"}

# Dataset paths
DATA_ROOT="/mnt/localssd/data"
METADATA_DIR="${DATA_ROOT}/LLaVA-OneVision-Data-processed/metadata"

# ============================================================
# Test mode: set TEST_MODE=1 to use only the first 1000 entries
# from CUSTOM_YAML (a single truncated JSONL) for faster iteration.
# Usage: TEST_MODE=1 bash scripts/lizard_scripts/train_stage1.sh
# ============================================================
TEST_MODE=${TEST_MODE:-0}
TEST_MODE_SAMPLES=${TEST_MODE_SAMPLES:-1000}

# ============================================================
# Auto-generate a temporary dataset YAML for all JSONL files
# ============================================================
CUSTOM_YAML=$(mktemp /tmp/llava_onevision_img_datasets_XXXXXX.yaml)
trap "rm -f ${CUSTOM_YAML}" EXIT

echo "---" > "${CUSTOM_YAML}"

if [ "${TEST_MODE}" = "1" ]; then
    # Merge all JSONL files, take the first TEST_MODE_SAMPLES lines,
    # write to a single temp JSONL, and register it as one dataset.
    TEST_JSONL=$(mktemp /tmp/llava_onevision_test_XXXXXX.jsonl)
    trap "rm -f ${CUSTOM_YAML} ${TEST_JSONL}" EXIT

    cat "${METADATA_DIR}"/*.jsonl | head -n "${TEST_MODE_SAMPLES}" > "${TEST_JSONL}"
    ACTUAL=$(wc -l < "${TEST_JSONL}")

    cat >> "${CUSTOM_YAML}" << YAML_ENTRY
'test_mode_data':
    _target_: llava.data.LLaVADataset
    data_path: ${TEST_JSONL}
    media_dir: ${DATA_ROOT}
YAML_ENTRY

    DATASET_NAMES=("test_mode_data")
    echo "TEST_MODE enabled: using ${ACTUAL} samples from ${TEST_JSONL}"
else
    DATASET_NAMES=()
    for jsonl_file in "${METADATA_DIR}"/*.jsonl; do
        stem=$(basename "${jsonl_file}" .jsonl)
        # Replace '+' with '_plus_' to avoid conflict with data_mixture separator
        name="${stem//+/_plus_}"
        DATASET_NAMES+=("${name}")
        # Quote the YAML key to safely handle parentheses and commas in filenames
        cat >> "${CUSTOM_YAML}" << YAML_ENTRY
'${name}':
    _target_: llava.data.LLaVADataset
    data_path: ${jsonl_file}
    media_dir: ${DATA_ROOT}
YAML_ENTRY
    done
    echo "Registered ${#DATASET_NAMES[@]} datasets from ${METADATA_DIR}"
fi

export VILA_DATASETS="default,${CUSTOM_YAML}"

# Build data_mixture: sort names and join with '+'
IFS=$'\n' SORTED_NAMES=($(sort <<<"${DATASET_NAMES[*]}"))
unset IFS
DATA_MIXTURE=$(IFS=+; echo "${SORTED_NAMES[*]}")

# ============================================================
# Single-GPU training setup (no SLURM dependency)
# ============================================================
RUN_NAME=${RUN_NAME:-$DEFAULT_RUN_NAME}
OUTPUT_DIR=${OUTPUT_DIR:-"runs/train/${RUN_NAME}"}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-$DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-$DEFAULT_GRADIENT_ACCUMULATION_STEPS}
LORA_R=${LORA_R:-$DEFAULT_LORA_R}
LORA_ALPHA=${LORA_ALPHA:-$DEFAULT_LORA_ALPHA}
LORA_DROPOUT=${LORA_DROPOUT:-$DEFAULT_LORA_DROPOUT}
DISTILL_TEMPERATURE=${DISTILL_TEMPERATURE:-1.0}

export WANDB_PROJECT=${WANDB_PROJECT:-"vila"}
export WANDB_DIR=${WANDB_DIR:-"${OUTPUT_DIR}"}
export WANDB_RUN_ID=${WANDB_RUN_ID:-"${RUN_NAME}"}
export WANDB_NAME=${WANDB_NAME:-"${RUN_NAME}"}
export WANDB_RESUME=${WANDB_RESUME:-"allow"}

export NCCL_IB_TIMEOUT=60
export TORCH_NCCL_BLOCKING_WAIT=0
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUTPUT_DIR}/model"

echo "MODEL_PATH   = ${MODEL_PATH}"
echo "TEACHER_PATH = ${TEACHER_MODEL_PATH}"
echo "DATASET_COUNT= ${#DATASET_NAMES[@]}"
echo "CUSTOM_YAML  = ${CUSTOM_YAML}"
echo "RUN_NAME     = ${RUN_NAME}"
echo "OUTPUT_DIR   = ${OUTPUT_DIR}"
echo "PER_DEVICE_BS= ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "GRAD_ACC_STP = ${GRADIENT_ACCUMULATION_STEPS}"
echo "LORA_R       = ${LORA_R}"
echo "LORA_ALPHA   = ${LORA_ALPHA}"
echo "LORA_DROPOUT = ${LORA_DROPOUT}"
echo "KD_TEMP      = ${DISTILL_TEMPERATURE}"

# ============================================================
# Launch training
# ============================================================
torchrun \
    --standalone --nnodes=1 --nproc_per_node=8 \
    llava/train/train_mem.py \
        --deepspeed scripts/zero3_gradient_clipping.json \
        --stage_type "stage1" \
        --total_time_limit -1 \
        --model_name_or_path ${MODEL_PATH} \
        --data_mixture "${DATA_MIXTURE}" \
        --vision_tower Efficient-Large-Model/paligemma-siglip-so400m-patch14-448 \
        --dynamic_s2 True \
        --s2_scales "448,896,1344" \
        --s2_max_split_size 448 \
        --s2_resize_output_to_scale_idx -1 \
        --mm_vision_select_feature cls_patch \
        --mm_projector mlp_downsample \
        --tune_vision_tower False \
        --tune_mm_projector False \
        --tune_language_model True \
        --distill_enable True \
        --teacher_model_name_or_path ${TEACHER_MODEL_PATH} \
        --distill_temperature ${DISTILL_TEMPERATURE} \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio dynamic_s2 \
        --max_grad_norm 5.0 \
        --bf16 True \
        --output_dir "${OUTPUT_DIR}/model" \
        --num_train_epochs 1 \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 100 \
        --save_total_limit 1 \
        --learning_rate 1e-3 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type constant \
        --logging_steps 1 \
        --model_max_length 8192 \
        --gradient_checkpointing True \
        --dataloader_num_workers 8 \
        --vflan_no_system_prompt True \
        --report_to wandb

        # --lora_enable True \
        # --lora_llm True \
        # --lora_vt False \
        # --lora_r ${LORA_R} \
        # --lora_alpha ${LORA_ALPHA} \
        # --lora_dropout ${LORA_DROPOUT} \