#!/bin/bash
# Train NVILA-8B on LLaVA-OneVision-Data-processed (image SFT stage)
# Run from the VILA root directory: bash scripts/train_nvila_8b_llava_onevision_img.sh

set -e

# ============================================================
# Configurable parameters (override via env vars)
# ============================================================
DEFAULT_RUN_NAME="nvila-8b-llava-onevision-img-sft"
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=108
DEFAULT_GRADIENT_ACCUMULATION_STEPS=4

# Starting checkpoint: pretrained NVILA-8B (stage3 output or HF model)
MODEL_PATH=${MODEL_PATH:-"Efficient-Large-Model/NVILA-8B"}

# Dataset paths
DATA_ROOT="/mnt/localssd/datasets"
METADATA_DIR="${DATA_ROOT}/LLaVA-OneVision-Data-processed/metadata"

# ============================================================
# Auto-generate a temporary dataset YAML for all JSONL files
# ============================================================
CUSTOM_YAML=$(mktemp /tmp/llava_onevision_img_datasets_XXXXXX.yaml)
trap "rm -f ${CUSTOM_YAML}" EXIT

echo "---" > "${CUSTOM_YAML}"

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
export VILA_DATASETS="default,${CUSTOM_YAML}"

# Build data_mixture: sort names and join with '+'
IFS=$'\n' SORTED_NAMES=($(sort <<<"${DATASET_NAMES[*]}"))
unset IFS
DATA_MIXTURE=$(IFS=+; echo "${SORTED_NAMES[*]}")

# ============================================================
# Common distributed training setup (reads SLURM env or defaults)
# ============================================================
source scripts/setups/train.sh

export NCCL_IB_TIMEOUT=60
export TORCH_NCCL_BLOCKING_WAIT=0
export TOKENIZERS_PARALLELISM=false

echo "MODEL_PATH   = ${MODEL_PATH}"
echo "DATASET_COUNT= ${#DATASET_NAMES[@]}"
echo "CUSTOM_YAML  = ${CUSTOM_YAML}"

# ============================================================
# Launch training
# ============================================================
torchrun \
    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
        --deepspeed scripts/zero2_gradient_clipping.json \
        --model_name_or_path ${MODEL_PATH} \
        --data_mixture "${DATA_MIXTURE}" \
        --vision_tower Efficient-Large-Model/paligemma-siglip-so400m-patch14-448 \
        --dynamic_s2 True \
        --s2_scales "448,896,1344" \
        --s2_max_split_size 448 \
        --s2_resize_output_to_scale_idx -1 \
        --mm_vision_select_feature cls_patch \
        --mm_projector mlp_downsample \
        --tune_vision_tower True \
        --tune_mm_projector True \
        --tune_language_model True \
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
        --learning_rate 1.5e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --model_max_length 8192 \
        --gradient_checkpointing True \
        --dataloader_num_workers 16 \
        --vflan_no_system_prompt True \
        --report_to wandb
