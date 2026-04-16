import argparse
import ast
import itertools
import json
import os
from collections import defaultdict

import torch
from datasets import load_dataset
from tqdm import tqdm

import llava
from llava import conversation as conversation_lib
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

CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def format_question(instance):
    """Build the text prompt for a single MMMU sample."""
    question = instance["question"]

    if instance["question_type"] == "multiple-choice":
        options = instance["options"]
        if isinstance(options, str):
            options = ast.literal_eval(options)
        choices_text = "\n".join(
            f"({CHOICE_LABELS[i]}) {opt}" for i, opt in enumerate(options)
        )
        prompt = (
            f"{question}\n"
            f"Options:\n{choices_text}\n"
            "Answer with the option's letter from the given choices directly."
        )
    else:
        prompt = f"{question}\nAnswer the question using a single word or phrase."

    return prompt


def collect_images(instance):
    """Return list of non-None PIL images from image_1..image_7 fields."""
    images = []
    for i in range(1, 8):
        img = instance.get(f"image_{i}")
        if img is not None:
            images.append(img)
    return images


def evaluate_outputs(outputs):
    """Compute overall and per-subject accuracy."""
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
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, required=True)
    parser.add_argument("--max-tiles", type=int, default=12)
    parser.add_argument("--generation-config", type=json.loads)
    parser.add_argument("--split", type=str, default="validation",
                        help="HuggingFace split name: validation | dev | test")
    parser.add_argument("--data-path", type=str, default="/mnt/localssd/data/eval/mmmu",
                        help="Local path to MMMU HuggingFace dataset dir")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    # Set up distributed environment
    dist.init()
    devices = range(dist.local_rank(), torch.cuda.device_count(), dist.local_size())
    torch.cuda.set_device(devices[0])

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_mode].copy()

    # Load model
    model = llava.load(args.model_path, model_base=args.model_base, devices=devices)
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

    # Collect all instances across all 30 subjects
    all_instances = []
    for subject in MMMU_SUBJECTS:
        ds = load_dataset(args.data_path, subject, split=args.split, trust_remote_code=True)
        for item in ds:
            item["subject"] = subject
            all_instances.append(item)

    # Chunk for this rank
    instances = all_instances[dist.rank() :: dist.size()]

    # Run inference
    outputs = []
    for instance in tqdm(instances, disable=not dist.is_main()):
        images = collect_images(instance)
        question = format_question(instance)

        # Build content list: interleave images referenced in question, then trailing images
        content = images + [question]
        response = model.generate_content(content, generation_config=generation_config)

        if instance["question_type"] == "multiple-choice":
            options = instance["options"]
            if isinstance(options, str):
                options = ast.literal_eval(options)
            all_choices = CHOICE_LABELS[: len(options)]
            index2ans = {CHOICE_LABELS[i]: opt for i, opt in enumerate(options)}
            pred = parse_choice(response, all_choices, index2ans)
        else:
            pred = response.strip()

        outputs.append({
            "id": instance["id"],
            "subject": instance["subject"],
            "question_type": instance["question_type"],
            "question": instance["question"],
            "answer": instance["answer"],
            "pred": pred,
            "response": response,
        })

    # Gather outputs across ranks
    if dist.size() > 1:
        outputs = dist.gather(outputs, dst=0)
        if not dist.is_main():
            return
        outputs = list(itertools.chain(*outputs))

    os.makedirs(args.output_dir, exist_ok=True)
    io.save(os.path.join(args.output_dir, "outputs.jsonl"), outputs)

    # Evaluate (skip for test split which has no labels)
    if args.split != "test":
        metrics = evaluate_outputs(outputs)
        io.save(os.path.join(args.output_dir, "metrics.json"), metrics)
        logger.info(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
