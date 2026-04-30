# Plan: Apply ViT-AdaLA-Style Three Stages to the NVILA LLM Decoder Without Modifying Existing Code

## Goal

Use the three-stage idea from ViT-AdaLA (`attention alignment -> feature alignment -> supervised fine-tuning`) on the NVILA LLM decoder. The implementation should be additive: create new scripts and helper modules, and do not modify existing repo files such as `llava/train/train.py`, `llava/train/llava_trainer.py`, or the existing `scripts/lizard_scripts`.

This is not a literal ViT implementation. The target is the autoregressive LLM decoder. The mapping is:

1. Attention alignment: align each linearized decoder self-attention output to the original softmax self-attention output.
2. Feature alignment: align the linearized decoder's hidden states and optionally logits to a frozen softmax teacher LLM.
3. Supervised fine-tuning: train the linearized decoder on normal SFT / VLM instruction data.

## Constraints

- Do not edit existing code.
- Do not modify existing `scripts/lizard_scripts/*.sh`.
- Do not modify `llava/train/*.py`.
- Add new files only.
- Reuse the existing Lizard/linear-attention behavior through public command-line flags wherever possible.
- If Python behavior must differ from existing training code, implement it in a new standalone runner script under `scripts/`.

## Current Repo Context

Existing LLM decoder linearization support:

- `scripts/lizard_scripts/train_stage1.sh`
  - Uses `llava/train/train_mem.py`.
  - Passes `--stage_type stage1`.
  - Passes `--distill_enable True`.
  - Patches LLM decoder self-attention with `LizardAttention` by default.
  - Computes per-layer attention-output MSE inside the existing trainer path.

- `scripts/lizard_scripts/train_stage2.sh`
  - Uses `--stage_type stage2`.
  - Uses `--attention_type lizard`.
  - Loads `--stage1_checkpoint_path`.
  - Applies LoRA and trains with normal token CE loss.
  - This is closer to supervised fine-tuning than to ViT-AdaLA Stage 2 feature alignment.

Existing reusable modules:

- `llava/train/linear_attn/lizard_attn.py`
- `llava/train/linear_attn/vanilla_linear_attn.py`
- `llava/train/linear_attn/monkey_patch.py`
- `llava/eval/load_lizard_model.py`

Important observation:

- Stage 1 already exists for the LLM decoder.
- Stage 3 mostly exists as current `scripts/lizard_scripts/train_stage2.sh`.
- The missing piece is a true ViT-AdaLA-style Stage 2: frozen teacher vs. linearized student hidden/logit alignment before final SFT.

## New Files

Create a new script directory:

- `scripts/adala_llm_decoder/`

Proposed new files:

- `scripts/adala_llm_decoder/train_stage1_attention_align.sh`
- `scripts/adala_llm_decoder/train_stage2_feature_align.sh`
- `scripts/adala_llm_decoder/train_stage3_sft.sh`
- `scripts/adala_llm_decoder/adala_stage2_feature_align.py`
- `scripts/adala_llm_decoder/README.md`

Optional later files:

- `scripts/adala_llm_decoder/eval_textvqa.sh`
- `scripts/adala_llm_decoder/eval_mmmu.sh`
- `scripts/adala_llm_decoder/merge_stage3_lora.sh`

## Stage 1: Decoder Attention Alignment

### Purpose

Train only the linear-attention parameters in each LLM decoder layer so the linear branch approximates the original softmax attention output.

### Implementation

Do not reimplement this stage. Create a wrapper script:

- `scripts/adala_llm_decoder/train_stage1_attention_align.sh`

The wrapper should be based on `scripts/lizard_scripts/train_stage1.sh`, but live in the new directory and call the existing training entrypoint with explicit names:

- `--stage_type stage1`
- `--distill_enable True`
- `--attention_type lizard` or `linear_attn`
- `--teacher_model_name_or_path ${TEACHER_MODEL_PATH}`

Default behavior:

- Use `attention_type=lizard`, matching current Lizard scripts.
- Support `ATTENTION_TYPE=linear_attn` for vanilla linear attention.
- Save to `runs/train/adala-llm-stage1-${ATTENTION_TYPE}`.

Trainable parameters:

- Existing code freezes the whole model and leaves the newly inserted attention-specific modules trainable.
- For `lizard`: train feature maps and gate parameters.
- For `linear_attn`: train parallel linear Q/K/V adapters.

Deliverable:

- A Stage 1 checkpoint containing trained linear-attention parameters.

## Stage 2: Decoder Feature and Logit Alignment

### Purpose

This is the key new stage. Train a linearized student decoder to match a frozen original softmax teacher decoder, reducing accumulated layer-wise approximation error before SFT.

### New Runner

Create:

- `scripts/adala_llm_decoder/adala_stage2_feature_align.py`

This script should be standalone and avoid changes to `llava/train/train.py` or `llava_trainer.py`.

High-level flow:

1. Load tokenizer and processor using the same NVILA builder utilities used by training/eval.
2. Load teacher model from `--teacher_model_name_or_path`.
3. Load student model from `--model_name_or_path`.
4. Patch the student LLM decoder attention with `LizardAttention` or `VanillaLinearAttention`.
5. Load Stage 1 attention weights from `--stage1_checkpoint_path`.
6. Freeze teacher.
7. Train student with hidden-state and optional logit distillation losses.
8. Save student checkpoint to `--output_dir`.

### Student Patching

Reuse existing modules without editing them:

- Import `apply_linear_attn_monkey_patches`.
- Import `LizardAttention`.
- Import `VanillaLinearAttention`.

Patch logic inside the new runner:

- Get `llm = model.get_llm()`.
- Get `layers = llm.model.layers`.
- Replace each `layer.self_attn` with selected attention class.
- Load only matching Stage 1 attention tensors:
  - for `lizard`: `feature_map_q`, `feature_map_k`, `gated_proj`
  - for `linear_attn`: `linear_q_proj`, `linear_k_proj`, `linear_v_proj`

This mirrors current Stage 2 behavior but stays in the new script.

### Loss

For the same multimodal/text batch:

- Teacher forward:
  - no grad
  - original softmax attention
  - `output_hidden_states=True`
  - optionally return logits

- Student forward:
  - linearized attention
  - `output_hidden_states=True`
  - optionally return logits

Primary loss:

```text
hidden_loss = MSE(student_hidden_states[-1], teacher_hidden_states[-1])
```

Optional stronger loss:

```text
layer_loss = mean over selected decoder layers:
    MSE(student_hidden_states[i], teacher_hidden_states[i])
```

Optional logit loss:

```text
logit_kl = KL(
    log_softmax(student_logits / T),
    softmax(teacher_logits / T)
) * T^2
```

Optional CE:

```text
ce_loss = standard language modeling CE on labels
```

Default combined objective:

```text
loss =
    hidden_weight * hidden_loss
  + layer_weight  * layer_loss
  + logit_weight  * logit_kl
  + ce_weight     * ce_loss
```

Recommended defaults:

- `hidden_weight=1.0`
- `layer_weight=0.0` initially
- `logit_weight=0.1`
- `ce_weight=0.0` for pure Stage 2 alignment
- `temperature=2.0`

Masking:

- Compute hidden/logit losses only on valid, non-padding tokens.
- For multimodal batches, include image token positions unless shape or label masking makes this unstable.
- Add an option `--align_label_tokens_only` to restrict losses to tokens with labels not equal to `IGNORE_INDEX`.

### Trainable Parameters

Initial conservative policy:

- Freeze vision tower.
- Freeze multimodal projector.
- Train the linearized LLM decoder attention modules.
- Optionally train LoRA on LLM in Stage 2 with `--lora_enable`.

Recommended first implementation:

- Train attention-specific linear modules only.
- Add `--train_lora True` later if Stage 2 loss plateaus.

### Data

Use the same LLaVA-OneVision processed data pattern from existing scripts:

- `DATA_ROOT=/mnt/localssd/data`
- `METADATA_DIR=${DATA_ROOT}/LLaVA-OneVision-Data-processed/metadata`

The new shell script should auto-generate the temporary YAML the same way as the existing Lizard scripts.

Support:

- `TEST_MODE=1`
- `TEST_MODE_SAMPLES=1000`

### Shell Entrypoint

Create:

- `scripts/adala_llm_decoder/train_stage2_feature_align.sh`

Responsibilities:

- Build dataset YAML.
- Set W&B/env variables.
- Launch with `torchrun`.
- Call `scripts/adala_llm_decoder/adala_stage2_feature_align.py`.

Important args:

- `--model_name_or_path`
- `--teacher_model_name_or_path`
- `--stage1_checkpoint_path`
- `--attention_type`
- `--data_mixture`
- `--output_dir`
- `--hidden_weight`
- `--logit_weight`
- `--temperature`
- `--model_max_length`

Output:

- `runs/train/adala-llm-stage2-feature-align-${ATTENTION_TYPE}/model`

## Stage 3: Supervised Fine-Tuning

### Purpose

Use the Stage 2 aligned linearized decoder for normal SFT/VLM training.

### Implementation

Create a wrapper:

- `scripts/adala_llm_decoder/train_stage3_sft.sh`

This should be based on existing `scripts/lizard_scripts/train_stage2.sh`, but with corrected naming:

- `STAGE2_CHECKPOINT_PATH` points to Stage 2 feature-aligned checkpoint.
- `STAGE1_CHECKPOINT_PATH` can still be accepted for fallback compatibility.
- `RUN_NAME=adala-llm-stage3-sft-${ATTENTION_TYPE}`.

Two implementation options:

1. Preferred initial option:
   - Call existing `llava/train/train_mem.py`.
   - Use `--stage_type stage2`.
   - Use `--attention_type ${ATTENTION_TYPE}`.
   - Use `--stage1_checkpoint_path ${STAGE2_CHECKPOINT_PATH}` if the Stage 2 checkpoint stores the same attention-module keys expected by existing loading logic.

2. Fallback option:
   - Add a standalone Stage 3 Python runner under `scripts/adala_llm_decoder/`.
   - Patch student attention and load Stage 2 weights manually, similar to Stage 2.
   - Then run standard CE training.

Start with option 1 to keep the implementation small.

## Evaluation

For evaluation, create wrappers first:

- `scripts/adala_llm_decoder/eval_textvqa.sh`
- `scripts/adala_llm_decoder/eval_mmmu.sh`

Use existing lizard eval loaders where possible. If Stage 2 checkpoints cannot be loaded by existing eval code, add a new script-local loader:

- `scripts/adala_llm_decoder/load_adala_llm_model.py`

This loader can be copied from the logic in `llava/eval/load_lizard_model.py`, but should live under `scripts/adala_llm_decoder/` to preserve the "no existing code modification" constraint.

## Verification Plan

Stage 1:

- Run `TEST_MODE=1 TEST_MODE_SAMPLES=32 scripts/adala_llm_decoder/train_stage1_attention_align.sh`.
- Confirm checkpoint contains expected attention-specific weights.

Stage 2:

- Run one tiny `torchrun` job with `TEST_MODE=1 TEST_MODE_SAMPLES=16`.
- Confirm:
  - teacher parameters have no gradients
  - student has patched linear attention
  - hidden loss is finite
  - logit KL is finite when enabled
  - one optimizer step updates only expected parameters

Stage 3:

- Run `TEST_MODE=1 TEST_MODE_SAMPLES=32 scripts/adala_llm_decoder/train_stage3_sft.sh`.
- Confirm standard CE loss runs without shape errors.

Numerical sanity:

- Stage 1 attention MSE should decrease on a tiny overfit subset.
- Stage 2 hidden MSE should decrease on a tiny overfit subset.
- Stage 3 CE should decrease or at least remain stable on a tiny overfit subset.

Regression:

- Existing `scripts/lizard_scripts` remain untouched.
- Existing training entrypoints remain untouched.
- Standard NVILA behavior is unchanged unless a new `scripts/adala_llm_decoder/*` script is explicitly used.

## Risks

- Duplicating teacher and student full NVILA models may exceed GPU memory.
  - Mitigation: start with small batch size, bf16, gradient checkpointing, and optionally load teacher on another device or use CPU/offload if needed.

- Hidden-state keys may differ depending on wrapper/model output format.
  - Mitigation: implement robust extraction for dict, tuple, and `ModelOutput`.

- Existing Stage 3 loader expects Stage 1-style attention keys, while Stage 2 may save a fuller student checkpoint.
  - Mitigation: save a filtered attention-only checkpoint from Stage 2 in addition to full model state.

- Multimodal batches may contain ignored labels for image/context tokens.
  - Mitigation: support both all-token hidden alignment and label-token-only alignment.

- Stage 2 pure hidden alignment may hurt generation if logits drift.
  - Mitigation: include optional logit KL with a small default weight.

## Implementation Order

1. Add `scripts/adala_llm_decoder/README.md`.
2. Add `train_stage1_attention_align.sh` wrapper.
3. Add `adala_stage2_feature_align.py`.
4. Add `train_stage2_feature_align.sh`.
5. Add `train_stage3_sft.sh` wrapper.
6. Run Stage 1 tiny smoke test or reuse an existing Stage 1 checkpoint.
7. Run Stage 2 tiny smoke test.
8. Run Stage 3 tiny smoke test.

