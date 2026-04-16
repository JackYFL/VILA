"""
Helper to load a Lizard stage2 checkpoint for evaluation.

Stage2 checkpoints are NOT standard VILA checkpoints: they are LoRA adapters
applied on top of a Lizard-patched NVILA base model. Loading requires:

  1. Load full multimodal base model (NVILA-8B).
  2. Apply LizardAttention monkey patches (patches decoder-layer forward).
  3. Replace every decoder-layer self_attn with LizardAttention.
  4. Load stage1 Lizard weights (feature_map_q, feature_map_k, gated_proj).
  5. Load non_lora_trainables (embed_tokens) from stage2 checkpoint.
  6. Apply LoRA adapter from stage2 checkpoint and merge.

Usage:
    from llava.eval.load_lizard_model import load_lizard_stage2_model
    model = load_lizard_stage2_model(
        base_model_path="Efficient-Large-Model/NVILA-8B",
        stage1_ckpt_path="runs/train/.../checkpoint-1500",
        stage2_ckpt_path="runs/train/.../checkpoint-20194",
        devices=[0, 1, ...],
    )
"""
import glob
import os

import torch


def load_lizard_stage2_model(
    base_model_path: str,
    stage1_ckpt_path: str,
    stage2_ckpt_path: str,
    devices=None,
):
    """Return a fully merged VILA model ready for eval."""
    import llava
    from llava.train.linear_attn import LizardAttention, apply_linear_attn_monkey_patches

    # ------------------------------------------------------------------
    # 1. Load base multimodal model
    # ------------------------------------------------------------------
    print(f"[Lizard] Loading base model from {base_model_path} ...")
    model = llava.load(base_model_path, devices=devices)

    # ------------------------------------------------------------------
    # 2. Apply decoder-layer monkey patches (handles 3-tuple attn return)
    # ------------------------------------------------------------------
    apply_linear_attn_monkey_patches()

    # ------------------------------------------------------------------
    # 3. Replace every self_attn with LizardAttention
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
    # 4. Load stage1 Lizard weights (feature_map_q/k and gated_proj base)
    # ------------------------------------------------------------------
    stage1_llm_dir = os.path.join(stage1_ckpt_path, "llm")
    if not os.path.isdir(stage1_llm_dir):
        stage1_llm_dir = stage1_ckpt_path
    print(f"[Lizard] Loading stage1 Lizard weights from {stage1_llm_dir} ...")

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

    _lizard_keys = ("feature_map_q.", "feature_map_k.", "gated_proj.")
    lizard_weights = {k: v for k, v in stage1_sd.items()
                      if any(lk in k for lk in _lizard_keys)}
    missing, unexpected = llm.load_state_dict(lizard_weights, strict=False)
    print(f"[Lizard] Loaded {len(lizard_weights)} Lizard tensors from stage1 "
          f"(missing: {len(missing)}, unexpected: {len(unexpected)}).")

    # ------------------------------------------------------------------
    # 5. Load non_lora_trainables (embed_tokens etc.) from stage2 ckpt
    # ------------------------------------------------------------------
    non_lora_path = os.path.join(stage2_ckpt_path, "non_lora_trainables.bin")
    if os.path.exists(non_lora_path):
        non_lora = torch.load(non_lora_path, map_location="cpu", weights_only=False)
        # Strip PEFT prefix: "base_model.model." → ""
        non_lora = {
            (k[len("base_model.model."):] if k.startswith("base_model.model.") else k): v
            for k, v in non_lora.items()
        }
        missing2, _ = model.load_state_dict(non_lora, strict=False)
        print(f"[Lizard] Loaded non_lora_trainables: {list(non_lora.keys())} "
              f"(missing: {len(missing2)}).")

    # ------------------------------------------------------------------
    # 6. Apply LoRA adapter from stage2 and merge in-place
    # ------------------------------------------------------------------
    from peft import PeftModel

    print(f"[Lizard] Applying LoRA from {stage2_ckpt_path} ...")
    model = PeftModel.from_pretrained(model, stage2_ckpt_path)
    print("[Lizard] Merging LoRA weights ...")
    model = model.merge_and_unload()
    model.eval()
    print("[Lizard] Model ready.")
    return model
