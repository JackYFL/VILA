"""
MMMU / MMMU-Pro eval for Lizard stage2 checkpoints.

Stage2 checkpoints cannot be loaded with plain llava.load() because they are
LoRA adapters on top of a Lizard-patched NVILA model.  Use this script instead
of mmmu.py when evaluating a stage2 Lizard checkpoint.

Usage:
    torchrun --nproc-per-node=N llava/eval/mmmu_lizard.py \
        --base-model  Efficient-Large-Model/NVILA-8B         \
        --stage1-ckpt runs/train/.../checkpoint-1500         \
        --stage2-ckpt runs/train/.../checkpoint-20194        \
        --conv-mode   hermes-2                               \
        --split       validation                             \
        --data-path   /mnt/localssd/data/eval/mmmu           \
        --output-dir  runs/eval/.../mmmu
"""
import argparse
import ast
import itertools
import json
import os
from collections import defaultdict

import torch
from datasets import load_dataset
from tqdm import tqdm

from llava import conversation as conversation_lib
from llava.eval.load_lizard_model import load_lizard_stage2_alone, load_lizard_stage2_model
from llava.eval.mmmu_utils.eval_utils import parse_choice
from llava.utils import distributed as dist
from llava.utils import io
from llava.utils.logging import logger

MMMU_SUBJECTS = [
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory",
    "Basic_Medical_Science", "Biology", "Chemistry", "Clinical_Medicine", "Computer_Science",
    "Design", "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics",
    "Energy_and_Power", "Finance", "Geography", "History", "Literature",
    "Manage", "Marketing", "Materials", "Math", "Mechanical_Engineering",
    "Music", "Pharmacy", "Physics", "Psychology", "Public_Health", "Sociology",
]

MMMU_PRO_CONFIGS = [
    "standard (4 options)",
    "standard (10 options)",
    "vision",
]

CHOICE_LABELS = [chr(ord("A") + i) for i in range(26)]


def format_question(instance, is_pro=False, is_vision=False):
    options = instance["options"]
    if isinstance(options, str):
        options = ast.literal_eval(options)
    choices_text = "\n".join(
        f"({CHOICE_LABELS[i]}) {opt}" for i, opt in enumerate(options)
    )

    if is_vision:
        prompt = (
            f"Options:\n{choices_text}\n"
            "Answer with the option's letter from the given choices directly."
        )
        return prompt

    question = instance["question"]

    if is_pro or instance.get("question_type") == "multiple-choice":
        prompt = (
            f"{question}\n"
            f"Options:\n{choices_text}\n"
            "Answer with the option's letter from the given choices directly."
        )
    else:
        prompt = f"{question}\nAnswer the question using a single word or phrase."

    return prompt


def collect_images(instance, is_vision=False):
    if is_vision:
        img = instance.get("image")
        return [img] if img is not None else []
    images = []
    for i in range(1, 8):
        img = instance.get(f"image_{i}")
        if img is not None:
            images.append(img)
    return images


def load_mmmu_instances(data_path, split):
    all_instances = []
    for subject in MMMU_SUBJECTS:
        ds = load_dataset(data_path, subject, split=split, trust_remote_code=True)
        for item in ds:
            item["subject"] = subject
            item["_pro"] = False
            item["_vision"] = False
            all_instances.append(item)
    return all_instances


def load_mmmu_pro_instances(data_path, configs=None):
    if configs is None:
        configs = MMMU_PRO_CONFIGS
    all_instances = []
    for cfg in configs:
        is_vision = cfg == "vision"
        ds = load_dataset(data_path, cfg, split="test", trust_remote_code=True)
        for idx, item in enumerate(ds):
            item["_pro"] = True
            item["_vision"] = is_vision
            item["_config"] = cfg
            if "id" not in item:
                item["id"] = f"{cfg}_{idx}"
            all_instances.append(item)
    return all_instances


def evaluate_outputs(outputs):
    correct = 0
    subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for out in outputs:
        gt = out["answer"]
        pred = out["pred"]
        is_correct = pred == gt
        correct += int(is_correct)
        subj = out["subject"]
        subject_stats[subj]["correct"] += int(is_correct)
        subject_stats[subj]["total"] += 1

    total = len(outputs)
    metrics = {
        "accuracy": round(correct / total * 100, 2) if total else 0.0,
        "correct": correct,
        "total": total,
        "subject": {
            subj: {
                "accuracy": round(s["correct"] / s["total"] * 100, 2),
                "correct": s["correct"],
                "total": s["total"],
            }
            for subj, s in sorted(subject_stats.items())
        },
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, required=True,
                        help="Path or HF ID of the base model (e.g. Efficient-Large-Model/NVILA-8B)")
    parser.add_argument("--stage1-ckpt", type=str, default=None,
                        help="Optional stage1 checkpoint dir. If omitted, the loader consolidates "
                             "stage2's DeepSpeed ZeRO shards (global_step*/) and loads everything "
                             "from stage2 alone via load_lizard_stage2_alone.")
    parser.add_argument("--stage2-ckpt", type=str, required=True,
                        help="Path to stage2 checkpoint dir (contains adapter_model.safetensors)")
    parser.add_argument("--conv-mode", type=str, required=True)
    parser.add_argument("--max-tiles", type=int, default=12)
    parser.add_argument("--generation-config", type=json.loads)
    parser.add_argument("--output-dir", type=str, required=True)

    # Standard MMMU options
    parser.add_argument("--split", type=str, default="validation",
                        help="HuggingFace split name: validation | dev | test")
    parser.add_argument("--data-path", type=str, default="/mnt/localssd/data/eval/mmmu",
                        help="Local path to MMMU HuggingFace dataset dir")

    # MMMU-Pro options
    parser.add_argument("--pro", action="store_true",
                        help="Evaluate on MMMU-Pro instead of standard MMMU")
    parser.add_argument("--pro-data-path", type=str, default="/mnt/localssd/data/eval/mmmu_pro",
                        help="Local path to MMMU-Pro HuggingFace dataset dir")
    parser.add_argument("--pro-configs", type=str, nargs="+", default=None,
                        help="MMMU-Pro configs to run (default: all three). "
                             "Choices: 'standard (4 options)' 'standard (10 options)' 'vision'")

    # Root-cause-2 fix (see textvqa_lizard.py): default to full-sequence mode.
    parser.add_argument(
        "--no-cache", dest="use_cache", action="store_false", default=False,
        help="Disable LinearGatedCache (full-sequence mode, matches training). This is the default.",
    )
    parser.add_argument(
        "--use-cache", dest="use_cache", action="store_true",
        help="Enable LinearGatedCache (fast recurrent inference).",
    )
    args = parser.parse_args()

    # Set up distributed environment
    dist.init()
    devices = range(dist.local_rank(), torch.cuda.device_count(), dist.local_size())
    torch.cuda.set_device(devices[0])

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_mode].copy()

    # Load Lizard stage2 model
    if args.stage1_ckpt:
        model = load_lizard_stage2_model(
            base_model_path=args.base_model,
            stage1_ckpt_path=args.stage1_ckpt,
            stage2_ckpt_path=args.stage2_ckpt,
            devices=devices,
        )
    else:
        model = load_lizard_stage2_alone(
            base_model_path=args.base_model,
            stage2_ckpt_path=args.stage2_ckpt,
            devices=devices,
        )

    # Configure tile counts
    model.config.min_tiles = 1
    model.config.max_tiles = args.max_tiles
    model.llm.config.min_tiles = 1
    model.llm.config.max_tiles = args.max_tiles

    # Adjust context length for high tile counts
    context_length = model.tokenizer.model_max_length
    if args.max_tiles > 12:
        context_length = max(context_length, int(args.max_tiles / 12.0 * 4096))
    model.config.model_max_length = context_length
    model.config.tokenizer_model_max_length = context_length
    model.llm.config.model_max_length = context_length
    model.llm.config.tokenizer_model_max_length = context_length
    model.tokenizer.model_max_length = context_length

    # Set up generation config
    generation_config = model.default_generation_config
    if args.generation_config is not None:
        generation_config.update(**args.generation_config)
    generation_config.use_cache = args.use_cache
    logger.info(f"use_cache={generation_config.use_cache} "
                f"({'recurrent/cache' if args.use_cache else 'full-sequence/no-cache'})")

    # Load instances
    if args.pro:
        all_instances = load_mmmu_pro_instances(args.pro_data_path, args.pro_configs)
        can_evaluate = True
    else:
        all_instances = load_mmmu_instances(args.data_path, args.split)
        can_evaluate = args.split != "test"

    # Chunk for this rank
    instances = all_instances[dist.rank() :: dist.size()]

    # Run inference
    outputs = []
    for instance in tqdm(instances, disable=not dist.is_main()):
        is_pro = instance.get("_pro", False)
        is_vision = instance.get("_vision", False)

        images = collect_images(instance, is_vision=is_vision)
        question = format_question(instance, is_pro=is_pro, is_vision=is_vision)

        content = images + [question]
        response = model.generate_content(content, generation_config=generation_config)

        is_mc = is_pro or instance.get("question_type") == "multiple-choice"
        if is_mc:
            options = instance["options"]
            if isinstance(options, str):
                options = ast.literal_eval(options)
            all_choices = CHOICE_LABELS[: len(options)]
            index2ans = {CHOICE_LABELS[i]: opt for i, opt in enumerate(options)}
            pred = parse_choice(response, all_choices, index2ans)
        else:
            pred = response.strip()

        out_item = {
            "id": instance.get("id", ""),
            "subject": instance.get("subject", ""),
            "answer": instance["answer"],
            "pred": pred,
            "response": response,
        }
        if is_pro:
            out_item["config"] = instance.get("_config", "")
        else:
            out_item["question_type"] = instance.get("question_type", "")
            out_item["question"] = instance.get("question", "")
        outputs.append(out_item)

    # Gather outputs across ranks
    if dist.size() > 1:
        outputs = dist.gather(outputs, dst=0)
        if not dist.is_main():
            return
        outputs = list(itertools.chain(*outputs))

    os.makedirs(args.output_dir, exist_ok=True)
    io.save(os.path.join(args.output_dir, "outputs.jsonl"), outputs)

    if can_evaluate:
        metrics = evaluate_outputs(outputs)

        if args.pro:
            config_stats = defaultdict(lambda: {"correct": 0, "total": 0})
            for out in outputs:
                cfg = out.get("config", "unknown")
                gt = out["answer"]
                pred = out["pred"]
                config_stats[cfg]["correct"] += int(pred == gt)
                config_stats[cfg]["total"] += 1
            metrics["config"] = {
                cfg: {
                    "accuracy": round(s["correct"] / s["total"] * 100, 2),
                    "correct": s["correct"],
                    "total": s["total"],
                }
                for cfg, s in sorted(config_stats.items())
            }

        io.save(os.path.join(args.output_dir, "metrics.json"), metrics)
        logger.info(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
