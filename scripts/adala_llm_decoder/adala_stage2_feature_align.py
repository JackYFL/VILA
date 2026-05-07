#!/usr/bin/env python3
"""AdaLA-style Stage 2 hidden/logit alignment for NVILA's LLM decoder.

This runner is intentionally additive. It does not modify llava/train/*.py.
"""

import glob
import math
import os
import copy
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from transformers import AutoConfig, HfArgumentParser, set_seed

from llava import conversation as conversation_lib
from llava.constants import IGNORE_INDEX
from llava.data import make_supervised_data_module
from llava.model import LlavaLlamaConfig, LlavaLlamaModel, LlavaTopDownLlamaConfig, LlavaTopDownLlamaModel
from llava.model.language_model.qllava_qllama import quantize_args_to_model_class
from llava.train.args import DataArguments, ModelArguments, TrainingArguments
from llava.train.callbacks.autoresume_callback import AutoResumeCallback
from llava.train.linear_attn import LizardAttention, VanillaLinearAttention, apply_linear_attn_monkey_patches
from llava.train.llava_trainer import LLaVATrainer
from llava.train.slurm_utils import TimeoutTerminateCallback
from llava.train.train import safe_save_model_for_hf_trainer, smart_tokenizer_and_embedding_resize
from llava.train.utils import get_checkpoint_path, mprint, prepare_config_for_training, vision_resolution_elevation


@dataclass
class AdaLAStage2Arguments:
    hidden_weight: float = field(default=1.0)
    layer_weight: float = field(default=0.0)
    logit_weight: float = field(default=0.1)
    ce_weight: float = field(default=0.0)
    temperature: float = field(default=2.0)
    align_label_tokens_only: bool = field(default=False)
    train_lora: bool = field(default=False)


def _build_model(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
    resume_path, continue_training = get_checkpoint_path(training_args.output_dir)
    if not continue_training:
        raise RuntimeError(f"Model appears complete under {training_args.output_dir}; refusing to retrain.")

    resume_from_checkpoint = bool(resume_path)
    if resume_from_checkpoint and not training_args.lora_enable:
        config = AutoConfig.from_pretrained(resume_path, trust_remote_code=True)
        config.resume_path = resume_path
        model_cls = eval(config.architectures[0])
    else:
        if model_args.ps3:
            model_cls = LlavaTopDownLlamaModel
            config = LlavaTopDownLlamaConfig.from_pretrained(
                model_args.model_name_or_path, resume=resume_from_checkpoint
            )
        else:
            if model_args.quantize_model in quantize_args_to_model_class:
                from llava.model.language_model.qllava_qllama import QLlavaLlamaModel

                model_cls = QLlavaLlamaModel
            else:
                if model_args.quantize_model != "false":
                    raise ValueError(f"{model_args.quantize_model} is not supported by this runner.")
                model_cls = LlavaLlamaModel
            config = LlavaLlamaConfig.from_pretrained(model_args.model_name_or_path, resume=resume_from_checkpoint)

        if getattr(config, "resume_path", None) is not None:
            config.resume_path = model_args.model_name_or_path

    prepare_config_for_training(config, model_args, training_args, data_args)
    model = model_cls(
        config=config,
        attn_implementation="flash_attention_2",
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
    )
    vision_resolution_elevation(model, config)
    model.llm.config.use_cache = False
    return model, resume_from_checkpoint


def _prepare_tokenizer_and_data_args(
    model,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
):
    tokenizer = model.tokenizer
    if tokenizer.bos_token is None:
        smart_tokenizer_and_embedding_resize({"bos_token": "[BOS]"}, tokenizer, model.llm)
    tokenizer.pad_token = tokenizer.unk_token
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize({"pad_token": "[PAD]"}, tokenizer, model.llm)

    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True
        model.config.num_video_frames = data_args.num_video_frames if data_args.num_video_frames is not None else 8
        model.config.fps = data_args.fps if hasattr(data_args, "fps") else 0.0
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.vision_tower_lr = training_args.vision_tower_lr
        if model_args.mm_use_im_start_end:
            raise AssertionError("mm_use_im_start_end is not supported by this runner.")
        if model_args.mm_use_im_patch_token:
            raise AssertionError("mm_use_im_patch_token is not supported by this runner.")

        model.config.num_time_tokens = data_args.num_time_tokens = model_args.num_time_tokens
        model.config.time_token_format = data_args.time_token_format = model_args.time_token_format
        if model_args.num_time_tokens > 0:
            time_tokens = [model.config.time_token_format.format(t=t) for t in range(model.config.num_time_tokens)]
            num_new_tokens = tokenizer.add_tokens(time_tokens)
            if num_new_tokens > 0:
                model.resize_token_embeddings(len(tokenizer))
            model.config.time_token_ids = tokenizer.convert_tokens_to_ids(time_tokens)
        else:
            model.config.time_token_ids = []
        model.config.soft_ce_std = model_args.soft_ce_std

        num_patches = model.get_vision_tower().num_patches
        downsample_rate = model.get_mm_projector().downsample_rate
        data_args.num_image_tokens = math.ceil(num_patches**0.5 / downsample_rate) ** 2

    data_args.s2_scales = list(map(int, model_args.s2_scales.split(",")))
    return tokenizer


def _get_llm_layers(model):
    llm = model.get_llm()
    if llm is None:
        raise ValueError("Expected a VILA model with an LLM, but model.get_llm() returned None.")
    llm_model = getattr(llm, "model", llm)
    layers = getattr(llm_model, "layers", None)
    if layers is None:
        raise ValueError("Cannot find decoder layers at model.get_llm().model.layers.")
    return llm, layers


def patch_student_attention(model, attention_type: str) -> int:
    attention_type = attention_type.lower()
    registry = {
        "lizard": LizardAttention,
        "linear_attn": VanillaLinearAttention,
    }
    if attention_type not in registry:
        raise ValueError(f"Unsupported attention_type={attention_type}; expected one of {list(registry)}")

    apply_linear_attn_monkey_patches()
    _, layers = _get_llm_layers(model)
    patched = 0
    for layer in layers:
        if hasattr(layer, "self_attn"):
            layer.self_attn = registry[attention_type](layer.self_attn)
            patched += 1
    if patched == 0:
        raise ValueError("No decoder self_attn modules were patched.")
    return patched


def _load_checkpoint_state(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    if not checkpoint_path:
        return {}

    ckpt_dir = os.path.join(checkpoint_path, "llm")
    if not os.path.isdir(ckpt_dir):
        ckpt_dir = checkpoint_path

    state = {}
    safetensors = sorted(glob.glob(os.path.join(ckpt_dir, "model*.safetensors")))
    if safetensors:
        from safetensors.torch import load_file

        for shard in safetensors:
            state.update(load_file(shard, device="cpu"))
        return state

    bins = sorted(glob.glob(os.path.join(ckpt_dir, "pytorch_model*.bin")))
    bins += sorted(glob.glob(os.path.join(ckpt_dir, "*.bin")))
    for shard in bins:
        loaded = torch.load(shard, map_location="cpu")
        if isinstance(loaded, dict) and "state_dict" in loaded:
            loaded = loaded["state_dict"]
        if isinstance(loaded, dict):
            state.update(loaded)
    return state


def load_stage1_attention_weights(model, checkpoint_path: str, attention_type: str) -> int:
    state = _load_checkpoint_state(checkpoint_path)
    if not state:
        raise ValueError(f"No model weights found under {checkpoint_path}")

    if attention_type == "linear_attn":
        whitelist = ("linear_q_proj.", "linear_k_proj.", "linear_v_proj.")
    elif attention_type == "lizard":
        whitelist = ("feature_map_q.", "feature_map_k.", "gated_proj.")
    else:
        raise ValueError(f"Unsupported attention_type={attention_type}")

    filtered = {k: v for k, v in state.items() if any(token in k for token in whitelist)}
    if not filtered:
        raise ValueError(f"No {attention_type} attention tensors found in {checkpoint_path}")
    missing, unexpected = model.get_llm().load_state_dict(filtered, strict=False)
    mprint(
        f"Loaded {len(filtered)} {attention_type} tensors from {checkpoint_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})."
    )
    return len(filtered)


def freeze_for_stage2(model, attention_type: str):
    model.requires_grad_(False)
    if attention_type == "linear_attn":
        tokens = ("linear_q_proj", "linear_k_proj", "linear_v_proj")
    else:
        tokens = ("feature_map_q", "feature_map_k", "gated_proj")
    for name, param in model.named_parameters():
        if any(token in name for token in tokens):
            param.requires_grad_(True)


def extract_hidden_and_logits(outputs):
    hidden_states = getattr(outputs, "hidden_states", None)
    logits = getattr(outputs, "logits", None)
    loss = getattr(outputs, "loss", None)
    if isinstance(outputs, dict):
        hidden_states = outputs.get("hidden_states", hidden_states)
        logits = outputs.get("logits", logits)
        loss = outputs.get("loss", loss)
    if hidden_states is None:
        raise ValueError("Model outputs do not include hidden_states. Pass output_hidden_states=True.")
    return hidden_states, logits, loss


def masked_mse(student: torch.Tensor, teacher: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    diff = (student.float() - teacher.float()).pow(2).mean(dim=-1)
    if mask is None:
        return diff.mean()
    mask = mask.to(device=diff.device, dtype=diff.dtype)
    return (diff * mask).sum() / mask.sum().clamp_min(1.0)


def masked_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    temperature: float,
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1) * (temperature**2)
    if mask is None:
        return kl.mean()
    mask = mask.to(device=kl.device, dtype=kl.dtype)
    return (kl * mask).sum() / mask.sum().clamp_min(1.0)


class AdaLAStage2Trainer(LLaVATrainer):
    def __init__(self, *args, teacher_model=None, distill_args: AdaLAStage2Arguments = None, **kwargs):
        super().__init__(*args, **kwargs)
        if teacher_model is None:
            raise ValueError("teacher_model is required")
        if distill_args is None:
            raise ValueError("distill_args is required")
        self.teacher_model = teacher_model
        self.distill_args = distill_args
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)

    @staticmethod
    def _unwrap(model):
        return getattr(model, "module", model)

    @staticmethod
    def _forward_inputs(inputs):
        forward_inputs = dict(inputs)
        if "media" in forward_inputs and isinstance(forward_inputs["media"], dict):
            forward_inputs["media"] = {
                key: (list(value) if isinstance(value, list) else value)
                for key, value in forward_inputs["media"].items()
            }
        if "media_config" in forward_inputs:
            forward_inputs["media_config"] = copy.deepcopy(forward_inputs["media_config"])
        forward_inputs["output_hidden_states"] = True
        forward_inputs["use_cache"] = False
        forward_inputs["packing"] = False
        return forward_inputs

    @staticmethod
    def _embed_with_student(model, inputs):
        model = AdaLAStage2Trainer._unwrap(model)
        media = inputs.get("media")
        if media is None:
            media = {}
        media_config = inputs.get("media_config")
        if media_config is None:
            media_config = defaultdict(dict)
        else:
            media_config = copy.deepcopy(media_config)

        return model._embed(
            inputs.get("input_ids"),
            media,
            media_config,
            inputs.get("labels"),
            inputs.get("attention_mask"),
        )

    @staticmethod
    def _llm_forward(model, inputs_embeds, labels, attention_mask):
        model = AdaLAStage2Trainer._unwrap(model)
        return model.get_llm()(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            use_cache=False,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch

        inputs_embeds, embedded_labels, embedded_attention_mask = self._embed_with_student(model, inputs)

        with torch.no_grad():
            teacher_outputs = self._llm_forward(
                self.teacher_model,
                inputs_embeds.detach(),
                embedded_labels,
                embedded_attention_mask,
            )
        student_outputs = self._llm_forward(model, inputs_embeds, embedded_labels, embedded_attention_mask)

        teacher_hidden, teacher_logits, _ = extract_hidden_and_logits(teacher_outputs)
        student_hidden, student_logits, student_ce = extract_hidden_and_logits(student_outputs)

        labels = embedded_labels
        attention_mask = embedded_attention_mask

        hidden_mask = attention_mask
        if self.distill_args.align_label_tokens_only and labels is not None:
            hidden_mask = labels.ne(IGNORE_INDEX)

        loss = torch.zeros((), device=student_hidden[-1].device, dtype=torch.float32)
        metrics = {}

        hidden_loss = masked_mse(student_hidden[-1], teacher_hidden[-1], hidden_mask)
        loss = loss + self.distill_args.hidden_weight * hidden_loss
        metrics["adala_hidden_loss"] = hidden_loss.detach()

        if self.distill_args.layer_weight > 0:
            layer_losses = []
            for student_layer, teacher_layer in zip(student_hidden[1:], teacher_hidden[1:]):
                layer_losses.append(masked_mse(student_layer, teacher_layer, hidden_mask))
            if layer_losses:
                layer_loss = torch.stack(layer_losses).mean()
                loss = loss + self.distill_args.layer_weight * layer_loss
                metrics["adala_layer_loss"] = layer_loss.detach()

        if self.distill_args.logit_weight > 0:
            if student_logits is None or teacher_logits is None:
                raise ValueError("logit_weight > 0 requires logits in model outputs.")
            logit_mask = labels.ne(IGNORE_INDEX) if labels is not None else attention_mask
            logit_loss = masked_kl(student_logits, teacher_logits, logit_mask, self.distill_args.temperature)
            loss = loss + self.distill_args.logit_weight * logit_loss
            metrics["adala_logit_kl"] = logit_loss.detach()

        if self.distill_args.ce_weight > 0:
            if student_ce is None:
                raise ValueError("ce_weight > 0 requires model loss in outputs.")
            loss = loss + self.distill_args.ce_weight * student_ce
            metrics["adala_ce_loss"] = student_ce.detach()

        self.log(metrics)
        return (loss, student_outputs) if return_outputs else loss


def save_attention_only_checkpoint(model, output_dir: str, attention_type: str):
    if attention_type == "linear_attn":
        tokens = ("linear_q_proj.", "linear_k_proj.", "linear_v_proj.")
    else:
        tokens = ("feature_map_q.", "feature_map_k.", "gated_proj.")
    state = {
        name: param.detach().cpu()
        for name, param in model.get_llm().state_dict().items()
        if any(token in name for token in tokens)
    }
    attention_dir = os.path.join(output_dir, "attention_only")
    os.makedirs(attention_dir, exist_ok=True)
    torch.save(state, os.path.join(attention_dir, "pytorch_model.bin"))
    mprint(f"Saved {len(state)} attention tensors to {attention_dir}")


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, AdaLAStage2Arguments))
    model_args, data_args, training_args, distill_args = parser.parse_args_into_dataclasses()

    if os.getenv("RUN_NAME") is not None:
        training_args.run_name = os.getenv("RUN_NAME")
    else:
        training_args.run_name = os.path.basename(training_args.output_dir.rstrip("/"))

    if distill_args.train_lora or training_args.lora_enable:
        raise NotImplementedError("LoRA in Stage 2 is not implemented here; use Stage 3 for LoRA SFT.")

    set_seed(training_args.seed)

    student, resume_from_checkpoint = _build_model(model_args, data_args, training_args)
    tokenizer = _prepare_tokenizer_and_data_args(student, model_args, data_args, training_args)

    distill_args.attention_type = getattr(training_args, "attention_type", "lizard")
    distill_args.stage1_checkpoint_path = getattr(training_args, "stage1_checkpoint_path", None)
    distill_args.teacher_model_name_or_path = getattr(
        training_args, "teacher_model_name_or_path", None
    ) or model_args.model_name_or_path

    teacher_model_args = replace(model_args, model_name_or_path=distill_args.teacher_model_name_or_path)
    teacher_training_args = replace(training_args, output_dir=os.path.join(training_args.output_dir, "_teacher_no_train"))
    teacher, _ = _build_model(teacher_model_args, data_args, teacher_training_args)
    _prepare_tokenizer_and_data_args(teacher, teacher_model_args, data_args, teacher_training_args)
    teacher.eval()
    teacher.requires_grad_(False)

    patched = patch_student_attention(student, distill_args.attention_type)
    mprint(f"Patched {patched} student decoder self-attn modules with {distill_args.attention_type}.")
    load_stage1_attention_weights(student, distill_args.stage1_checkpoint_path, distill_args.attention_type)
    freeze_for_stage2(student, distill_args.attention_type)

    if training_args.gradient_checkpointing and hasattr(student.llm, "enable_input_require_grads"):
        student.llm.enable_input_require_grads()

    if training_args.bits == 16:
        if training_args.bf16:
            student.to(torch.bfloat16)
            teacher.to(torch.bfloat16)
        if training_args.fp16:
            student.to(torch.float16)
            teacher.to(torch.float16)

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    mprint(f"Stage2 trainable params: {trainable:,d} / {total:,d} ({100 * trainable / total:.4f}%)")

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, training_args=training_args)
    callbacks = [
        AutoResumeCallback(),
        TimeoutTerminateCallback(
            total_time_limit=training_args.total_time_limit if training_args.total_time_limit > 0 else 10**9
        ),
    ]

    trainer = AdaLAStage2Trainer(
        model=student,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=callbacks,
        teacher_model=teacher,
        distill_args=distill_args,
        **data_module,
    )

    mprint("length of dataloader:", len(trainer.get_train_dataloader()), len(trainer.train_dataset))
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_state()

    student.llm.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    if training_args.local_rank in (0, -1):
        save_attention_only_checkpoint(student, training_args.output_dir, distill_args.attention_type)


if __name__ == "__main__":
    main()
