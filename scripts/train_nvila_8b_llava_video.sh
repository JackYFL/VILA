#!/bin/bash
# Train NVILA-8B on LLaVA-Video-178K-processed (video SFT stage)
# Run from the VILA root directory: bash scripts/train_nvila_8b_llava_onevision.sh

set -e

# ============================================================
# Configurable parameters (override via env vars)
# ============================================================
DEFAULT_RUN_NAME="nvila-8b-llava-video-178k-sft"
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=256
DEFAULT_GRADIENT_ACCUMULATION_STEPS=8

# Starting checkpoint: pretrained NVILA-8B (stage3 output or HF model)
MODEL_PATH=${MODEL_PATH:-"Efficient-Large-Model/NVILA-8B"}

# Dataset paths
DATA_DIR="/mnt/localssd/datasets/LLaVA-Video-178K-processed"
JSONL_FILE="${DATA_DIR}/llava_video_178k_train.jsonl"
MEDIA_DIR="${DATA_DIR}/videos"

# ============================================================
# Register dataset via a temporary YAML
# ============================================================
CUSTOM_YAML=$(mktemp /tmp/llava_video_178k_datasets_XXXXXX.yaml)
trap "rm -f ${CUSTOM_YAML}" EXIT

cat > "${CUSTOM_YAML}" << EOF
---
llava-video-178k:
    _target_: llava.data.LLaVADataset
    data_path: ${JSONL_FILE}
    media_dir: ${MEDIA_DIR}
    is_video: true
EOF

export VILA_DATASETS="default,${CUSTOM_YAML}"
DATA_MIXTURE="llava-video-178k"

echo "JSONL_FILE  = ${JSONL_FILE}"
echo "MEDIA_DIR   = ${MEDIA_DIR}"

# ============================================================
# Common distributed training setup (reads SLURM env or defaults)
# ============================================================
source scripts/setups/train.sh

export NCCL_IB_TIMEOUT=31

echo "MODEL_PATH  = ${MODEL_PATH}"
echo "DATA_MIXTURE= ${DATA_MIXTURE}"

# ============================================================
# Launch training
# ============================================================
torchrun \
    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
        --deepspeed scripts/zero3_gradient_clipping.json \
        --model_name_or_path ${MODEL_PATH} \
        --data_mixture "${DATA_MIXTURE}" \
        --vision_tower Efficient-Large-Model/paligemma-siglip-so400m-patch14-448 \
        --mm_vision_select_feature cls_patch \
        --mm_projector mlp_downsample_2x2_fix \
        --tune_vision_tower True \
        --tune_mm_projector True \
        --tune_language_model True \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio resize \
        --video_encoder '{"_target_": "llava.model.encoders.TSPVideoEncoder", "pool_sizes": [[8, 1, 1]]}' \
        --num_video_frames 256 \
        --num_time_tokens 100 \
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
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --model_max_length 16384 \
        --gradient_checkpointing True \
        --dataloader_num_workers 16 \
        --vflan_no_system_prompt True \
        --report_to wandb
