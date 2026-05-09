#!/bin/bash
# AdaLA-style Stage 2 for the NVILA LLM decoder:
# hidden/logit alignment between a frozen softmax teacher and a linearized student.

set -e

DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE=1
DEFAULT_GRADIENT_ACCUMULATION_STEPS=8

MODEL_PATH=${MODEL_PATH:-"Efficient-Large-Model/NVILA-8B"}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-"${MODEL_PATH}"}
ATTENTION_TYPE=${ATTENTION_TYPE:-"lizard"}
STAGE1_CHECKPOINT_PATH=${STAGE1_CHECKPOINT_PATH:-"runs/train/adala-llm-stage1-${ATTENTION_TYPE}/model/checkpoint-1500"}

DATA_ROOT=${DATA_ROOT:-"/mnt/localssd/data"}
METADATA_DIR=${METADATA_DIR:-"${DATA_ROOT}/LLaVA-OneVision-Data-processed/metadata"}

TEST_MODE=${TEST_MODE:-0}
TEST_MODE_SAMPLES=${TEST_MODE_SAMPLES:-1000}

CUSTOM_YAML=$(mktemp /tmp/adala_llm_stage2_datasets_XXXXXX.yaml)
trap "rm -f ${CUSTOM_YAML} ${TEST_JSONL:-}" EXIT

echo "---" > "${CUSTOM_YAML}"

if [ "${TEST_MODE}" = "1" ]; then
    TEST_JSONL=$(mktemp /tmp/adala_llm_stage2_test_XXXXXX.jsonl)
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
        name="${stem//+/_plus_}"
        DATASET_NAMES+=("${name}")
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

IFS=$'\n' SORTED_NAMES=($(sort <<<"${DATASET_NAMES[*]}"))
unset IFS
DATA_MIXTURE=$(IFS=+; echo "${SORTED_NAMES[*]}")

RUN_NAME=${RUN_NAME:-"adala-llm-stage2-feature-align-${ATTENTION_TYPE}"}
OUTPUT_DIR=${OUTPUT_DIR:-"runs/train/${RUN_NAME}"}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-$DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-$DEFAULT_GRADIENT_ACCUMULATION_STEPS}

export WANDB_PROJECT=${WANDB_PROJECT:-"vila"}
export WANDB_DIR=${WANDB_DIR:-"${OUTPUT_DIR}"}
export WANDB_RUN_ID=${WANDB_RUN_ID:-"${RUN_NAME}"}
export WANDB_NAME=${WANDB_NAME:-"${RUN_NAME}"}
export WANDB_RESUME=${WANDB_RESUME:-"allow"}
export RUN_NAME

export NCCL_IB_TIMEOUT=60
export TORCH_NCCL_BLOCKING_WAIT=0
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUTPUT_DIR}/model"

echo "MODEL_PATH        = ${MODEL_PATH}"
echo "TEACHER_PATH      = ${TEACHER_MODEL_PATH}"
echo "ATTENTION_TYPE    = ${ATTENTION_TYPE}"
echo "STAGE1_CHECKPOINT = ${STAGE1_CHECKPOINT_PATH}"
echo "DATASET_COUNT     = ${#DATASET_NAMES[@]}"
echo "OUTPUT_DIR        = ${OUTPUT_DIR}"

torchrun \
    --standalone --nnodes=1 --nproc_per_node=${NPROC_PER_NODE:-8} \
    scripts/adala_llm_decoder/adala_stage2_feature_align.py \
        --deepspeed ${DEEPSPEED_CONFIG:-scripts/zero3_gradient_clipping.json} \
        --model_name_or_path ${MODEL_PATH} \
        --teacher_model_name_or_path ${TEACHER_MODEL_PATH} \
        --stage1_checkpoint_path "${STAGE1_CHECKPOINT_PATH}" \
        --attention_type "${ATTENTION_TYPE}" \
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
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio dynamic_s2 \
        --max_grad_norm 5.0 \
        --bf16 True \
        --output_dir "${OUTPUT_DIR}/model" \
        --num_train_epochs ${NUM_TRAIN_EPOCHS:-1} \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps ${SAVE_STEPS:-100} \
        --save_total_limit 1 \
        --learning_rate ${LEARNING_RATE:-2e-4} \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --model_max_length ${MODEL_MAX_LENGTH:-8192} \
        --gradient_checkpointing True \
        --dataloader_num_workers ${DATALOADER_NUM_WORKERS:-8} \
        --vflan_no_system_prompt True \
        --hidden_weight ${HIDDEN_WEIGHT:-1.0} \
        --layer_weight ${LAYER_WEIGHT:-0.0} \
        --logit_weight ${LOGIT_WEIGHT:-0.1} \
        --ce_weight ${CE_WEIGHT:-0.0} \
        --temperature ${DISTILL_TEMPERATURE:-2.0} \
        --report_to ${REPORT_TO:-wandb}
