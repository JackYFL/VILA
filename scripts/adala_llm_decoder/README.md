# AdaLA-Style LLM Decoder Training

This directory adds an additive, script-only implementation of an AdaLA-style
three-stage recipe for NVILA's LLM decoder. Existing training code and existing
`scripts/lizard_scripts` files are intentionally left untouched.

## Stages

1. `train_stage1_attention_align.sh`
   - Wraps the existing decoder attention-alignment path.
   - Uses `--stage_type stage1` and `--distill_enable True`.
   - Trains linear-attention parameters to match softmax attention outputs.

2. `train_stage2_feature_align.sh`
   - Runs `adala_stage2_feature_align.py`.
   - Loads a frozen softmax teacher and a linearized student.
   - Aligns decoder hidden states and optionally logits.

3. `train_stage3_sft.sh`
   - Wraps the existing Stage 2/SFT path with clearer naming.
   - Uses the feature-aligned checkpoint as initialization for final SFT.

## Quick Smoke Tests

```bash
TEST_MODE=1 TEST_MODE_SAMPLES=32 bash scripts/adala_llm_decoder/train_stage1_attention_align.sh
TEST_MODE=1 TEST_MODE_SAMPLES=16 bash scripts/adala_llm_decoder/train_stage2_feature_align.sh
TEST_MODE=1 TEST_MODE_SAMPLES=32 bash scripts/adala_llm_decoder/train_stage3_sft.sh
```

Set `ATTENTION_TYPE=linear_attn` to use the vanilla linear-attention path
instead of `lizard`.

