"""
Helpers to load a Lizard stage2 checkpoint for evaluation.

Two loaders are provided:

1. load_lizard_stage2_alone (preferred):
   Loads everything from the stage2 checkpoint alone, by consolidating the
   DeepSpeed ZeRO-3 shards in <stage2>/global_step*/ via zero_to_fp32.py into
   a single state dict that contains ALL trained params — LoRA deltas,
   feature_map_q/k, gated_proj base + LoRA, embed_tokens, and all the frozen
   base-model weights.  No stage1 checkpoint is needed.

2. load_lizard_stage2_model (legacy):
   The original 6-step path that also requires a stage1 checkpoint.  Kept for
   backwards compatibility with workflows that don't have the DeepSpeed state
   (e.g. after pruning global_step*/ to save disk).

Usage:
    from llava.eval.load_lizard_model import load_lizard_stage2_alone
    model = load_lizard_stage2_alone(
        base_model_path="Efficient-Large-Model/NVILA-8B",
        stage2_ckpt_path="runs/train/.../checkpoint-20194",
        devices=[0, 1, ...],
    )
"""
import glob
import os
import subprocess
import sys
from typing import Optional

import torch


def _consolidate_deepspeed_shards(stage2_ckpt_path: str, output_path: str) -> None:
    """Run the checkpoint's bundled zero_to_fp32.py to merge ZeRO shards."""
    z2f = os.path.join(stage2_ckpt_path, "zero_to_fp32.py")
    if not os.path.isfile(z2f):
        raise FileNotFoundError(
            f"zero_to_fp32.py not found at {z2f}.  This checkpoint may have been "
            f"saved without DeepSpeed, or global_step*/ was pruned.  Fall back to "
            f"load_lizard_stage2_model() with an explicit stage1 checkpoint."
        )
    print(f"[Lizard] Consolidating ZeRO shards → {output_path}")
    subprocess.check_call([sys.executable, z2f, stage2_ckpt_path, output_path])


def load_lizard_stage2_alone(
    base_model_path: str,
    stage2_ckpt_path: str,
    devices=None,
    consolidated_path: Optional[str] = None,
    force_reconsolidate: bool = False,
):
    """Load a Lizard stage2 checkpoint without needing stage1.

    The stage2 DeepSpeed state in <stage2>/global_step*/ contains the full set
    of model parameters at end-of-stage2, including Lizard-specific params
    (feature_map_q/k, gated_proj) that live outside adapter_model.safetensors
    and non_lora_trainables.bin.  We consolidate those shards into a single
    state dict and load it into a PEFT-wrapped Lizard-patched NVILA model.

    Args:
        base_model_path:    Path or HF ID of the base NVILA model (provides
                            vision_tower / mm_projector / tokenizer / config).
        stage2_ckpt_path:   Path to the stage2 checkpoint directory.
        devices:            Device range for llava.load.
        consolidated_path:  Where to read/cache the merged state dict.  Default
                            is <stage2>/consolidated.bin.  Roughly 15-16 GB for
                            a 7B-class model in fp32.
        force_reconsolidate: Re-run zero_to_fp32.py even if the cache exists.

    Returns:
        A fully merged VILA model ready for eval.
    """
    import llava
    from llava.train.linear_attn import LizardAttention, apply_linear_attn_monkey_patches
    from peft import PeftModel

    if consolidated_path is None:
        consolidated_path = os.path.join(stage2_ckpt_path, "consolidated.bin")

    # ------------------------------------------------------------------
    # 1. Consolidate DeepSpeed ZeRO shards (cached on disk between runs)
    # ------------------------------------------------------------------
    if force_reconsolidate or not os.path.exists(consolidated_path):
        _consolidate_deepspeed_shards(stage2_ckpt_path, consolidated_path)
    else:
        print(f"[Lizard] Reusing cached consolidated state dict: {consolidated_path}")

    # ------------------------------------------------------------------
    # 2. Load base multimodal model (only for vision_tower / mm_projector /
    #    tokenizer / config; all LLM weights will be overwritten below).
    # ------------------------------------------------------------------
    print(f"[Lizard] Loading base model from {base_model_path} ...")
    model = llava.load(base_model_path, devices=devices)

    # ------------------------------------------------------------------
    # 3. Apply Lizard monkey patches (decoder-layer forward handles 3-tuple attn)
    # ------------------------------------------------------------------
    apply_linear_attn_monkey_patches()

    # ------------------------------------------------------------------
    # 4. Replace every self_attn with LizardAttention (creates fresh
    #    feature_map_q/k and gated_proj; these will be overwritten in step 6)
    # ------------------------------------------------------------------
    llm = model.get_llm()
    llm_model = getattr(llm, "model", llm)
    layers = getattr(llm_model, "layers", None)
    if layers is None:
        raise ValueError("Cannot find .model.layers in the LLM.")

    patched = 0
    for layer in layers:
        if hasattr(layer, "self_attn"):
            layer.self_attn = LizardAttention(layer.self_attn)
            patched += 1
    print(f"[Lizard] Patched {patched} self_attn modules with LizardAttention.")

    # ------------------------------------------------------------------
    # 5. Wrap in PEFT (reads adapter_config.json, wraps LoRA targets,
    #    populates lora_A/lora_B from adapter_model.safetensors)
    # ------------------------------------------------------------------
    print(f"[Lizard] Wrapping with PEFT adapter from {stage2_ckpt_path} ...")
    model = PeftModel.from_pretrained(model, stage2_ckpt_path)

    # ------------------------------------------------------------------
    # 6. Overwrite all parameters from the consolidated state dict.
    #    Keys in consolidated.bin already use the PEFT prefix
    #    'base_model.model.' and include both base_layer and lora_A/B
    #    structures, so load_state_dict matches directly with strict=False.
    #    This supersedes what PEFT.from_pretrained populated and also
    #    subsumes non_lora_trainables (embed_tokens).
    # ------------------------------------------------------------------
    print(f"[Lizard] Loading consolidated weights from {consolidated_path} ...")
    consolidated = torch.load(consolidated_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(consolidated, strict=False)
    print(f"[Lizard] Loaded {len(consolidated)} tensors "
          f"(missing: {len(missing)}, unexpected: {len(unexpected)}).")

    # ------------------------------------------------------------------
    # 7. Merge LoRA into base and drop the PEFT wrapper
    # ------------------------------------------------------------------
    print("[Lizard] Merging LoRA weights ...")
    model = model.merge_and_unload()
    model.eval()
    print("[Lizard] Model ready.")
    return model


def load_lizard_stage2_model(
    base_model_path: str,
    stage1_ckpt_path: str,
    stage2_ckpt_path: str,
    devices=None,
):
    """Legacy loader that requires a stage1 checkpoint.

    Prefer load_lizard_stage2_alone() when the stage2 DeepSpeed state is still
    available.  This function is kept for checkpoints where global_step*/ was
    pruned to save disk.

    Returns a fully merged VILA model ready for eval.
    """
    import llava
    from llava.train.linear_attn import LizardAttention, apply_linear_attn_monkey_patches

    print(f"[Lizard] Loading base model from {base_model_path} ...")
    model = llava.load(base_model_path, devices=devices)

    apply_linear_attn_monkey_patches()

    llm = model.get_llm()
    llm_model = getattr(llm, "model", llm)
    layers = getattr(llm_model, "layers", None)
    if layers is None:
        raise ValueError("Cannot find .model.layers in the LLM.")

    patched = 0
    for layer in layers:
        if hasattr(layer, "self_attn"):
            layer.self_attn = LizardAttention(layer.self_attn)
            patched += 1
    print(f"[Lizard] Patched {patched} self_attn modules with LizardAttention.")

    stage1_llm_dir = os.path.join(stage1_ckpt_path, "llm")
    if not os.path.isdir(stage1_llm_dir):
        stage1_llm_dir = stage1_ckpt_path
    print(f"[Lizard] Loading stage1 LLM weights from {stage1_llm_dir} ...")

    stage1_sd = {}
    st_shards = sorted(glob.glob(os.path.join(stage1_llm_dir, "model*.safetensors")))
    if st_shards:
        from safetensors.torch import load_file as st_load
        for shard in st_shards:
            stage1_sd.update(st_load(shard, device="cpu"))
    else:
        for shard in sorted(glob.glob(os.path.join(stage1_llm_dir, "pytorch_model*.bin"))):
            stage1_sd.update(torch.load(shard, map_location="cpu"))

    if not stage1_sd:
        raise ValueError(f"No model weights found in {stage1_llm_dir}")

    missing, unexpected = llm.load_state_dict(stage1_sd, strict=False)
    print(f"[Lizard] Loaded {len(stage1_sd)} tensors from stage1 "
          f"(missing: {len(missing)}, unexpected: {len(unexpected)}).")

    non_lora_path = os.path.join(stage2_ckpt_path, "non_lora_trainables.bin")
    if os.path.exists(non_lora_path):
        non_lora = torch.load(non_lora_path, map_location="cpu", weights_only=False)
        non_lora = {
            (k[len("base_model.model."):] if k.startswith("base_model.model.") else k): v
            for k, v in non_lora.items()
        }
        missing2, _ = model.load_state_dict(non_lora, strict=False)
        print(f"[Lizard] Loaded non_lora_trainables: {list(non_lora.keys())} "
              f"(missing: {len(missing2)}).")

    from peft import PeftModel

    print(f"[Lizard] Applying LoRA from {stage2_ckpt_path} ...")
    model = PeftModel.from_pretrained(model, stage2_ckpt_path)
    print("[Lizard] Merging LoRA weights ...")
    model = model.merge_and_unload()
    model.eval()
    print("[Lizard] Model ready.")
    return model
