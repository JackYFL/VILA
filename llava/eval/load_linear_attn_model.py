"""
Helper to load a vanilla-linear-attention stage2 checkpoint for evaluation.

Mirrors ``load_lizard_stage2_alone`` in ``load_lizard_model.py`` but wraps each
decoder layer with ``VanillaLinearAttention`` instead of ``LizardAttention``.
"""
import os
from typing import Optional

import torch

from llava.eval.load_lizard_model import _consolidate_deepspeed_shards


def load_linear_attn_stage2_alone(
    base_model_path: str,
    stage2_ckpt_path: str,
    devices=None,
    consolidated_path: Optional[str] = None,
    force_reconsolidate: bool = False,
):
    """Load a vanilla-linear-attention stage2 checkpoint without a stage1.

    The stage2 DeepSpeed state in ``<stage2>/global_step*/`` contains the full
    set of end-of-stage2 parameters, including the feature_map_q/k weights
    that live outside adapter_model.safetensors and non_lora_trainables.bin.
    We consolidate those shards into a single state dict and load it into a
    PEFT-wrapped NVILA model whose self_attn modules have been replaced with
    ``VanillaLinearAttention``.

    Returns a fully merged VILA model ready for eval.
    """
    import llava
    from llava.train.linear_attn import (
        VanillaLinearAttention,
        apply_linear_attn_monkey_patches,
    )
    from peft import PeftModel

    if consolidated_path is None:
        consolidated_path = os.path.join(stage2_ckpt_path, "consolidated.bin")

    # 1. Consolidate DeepSpeed ZeRO shards (cached on disk between runs).
    if force_reconsolidate or not os.path.exists(consolidated_path):
        _consolidate_deepspeed_shards(stage2_ckpt_path, consolidated_path)
    else:
        print(f"[LinearAttn] Reusing cached consolidated state dict: {consolidated_path}")

    # 2. Load base multimodal model (for vision_tower / mm_projector / tokenizer / config).
    print(f"[LinearAttn] Loading base model from {base_model_path} ...")
    model = llava.load(base_model_path, devices=devices)

    # 3. Apply monkey patches (decoder-layer forward handles 3-tuple attn return).
    apply_linear_attn_monkey_patches()

    # 4. Replace every self_attn with VanillaLinearAttention (feature maps created
    #    here will be overwritten by the consolidated state dict in step 6).
    llm = model.get_llm()
    llm_model = getattr(llm, "model", llm)
    layers = getattr(llm_model, "layers", None)
    if layers is None:
        raise ValueError("Cannot find .model.layers in the LLM.")

    patched = 0
    for layer in layers:
        if hasattr(layer, "self_attn"):
            layer.self_attn = VanillaLinearAttention(layer.self_attn)
            patched += 1
    print(f"[LinearAttn] Patched {patched} self_attn modules with VanillaLinearAttention.")

    # 5. Wrap in PEFT (reads adapter_config.json, wraps LoRA targets, populates
    #    lora_A/lora_B from adapter_model.safetensors).
    print(f"[LinearAttn] Wrapping with PEFT adapter from {stage2_ckpt_path} ...")
    model = PeftModel.from_pretrained(model, stage2_ckpt_path)

    # 6. Overwrite all parameters from the consolidated state dict.
    print(f"[LinearAttn] Loading consolidated weights from {consolidated_path} ...")
    consolidated = torch.load(consolidated_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(consolidated, strict=False)
    print(f"[LinearAttn] Loaded {len(consolidated)} tensors "
          f"(missing: {len(missing)}, unexpected: {len(unexpected)}).")

    # 7. Merge LoRA into base and drop the PEFT wrapper.
    print("[LinearAttn] Merging LoRA weights ...")
    model = model.merge_and_unload()
    model.eval()
    print("[LinearAttn] Model ready.")
    return model
