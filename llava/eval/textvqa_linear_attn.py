"""
TextVQA eval for vanilla-linear-attention stage2 checkpoints.

Stage2 checkpoints cannot be loaded with plain llava.load() because they are
LoRA adapters on top of a VanillaLinearAttention-patched NVILA model.  Use
this script instead of textvqa.py when evaluating a vanilla-LA stage2
checkpoint.

Usage:
    torchrun --nproc-per-node=N llava/eval/textvqa_linear_attn.py \
        --base-model  Efficient-Large-Model/NVILA-8B         \
        --stage2-ckpt runs/train/.../checkpoint-XXXXX        \
        --conv-mode   hermes-2                               \
        --output-dir  runs/eval/.../textvqa
"""
import argparse
import itertools
import json
import os
import re

import torch
from PIL import Image
from tqdm import tqdm

from llava import conversation as conversation_lib
from llava.data.builder import DATASETS
from llava.eval.load_linear_attn_model import load_linear_attn_stage2_alone
from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
from llava.utils import distributed as dist
from llava.utils import io
from llava.utils.logging import logger


def prompt_processor(prompt):
    if prompt.startswith("OCR tokens: "):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        question = match.group(1)
    elif "Reference OCR token: " in prompt and len(prompt.split("\n")) == 3:
        if prompt.startswith("Reference OCR token:"):
            question = prompt.split("\n")[1]
        else:
            question = prompt.split("\n")[0]
    elif len(prompt.split("\n")) == 2:
        question = prompt.split("\n")[0]
    else:
        assert False

    return question.lower()


def eval_single(outputs, answers):
    answers = answers["data"]
    answers = {(annotation["image_id"], annotation["question"].lower()): annotation
               for annotation in answers}

    pred_list = []
    for result in outputs:
        annotation = answers[(result["question_id"], prompt_processor(result["prompt"]))]
        pred_list.append({
            "pred_answer": result["text"],
            "gt_answers": annotation["answers"],
        })

    evaluator = TextVQAAccuracyEvaluator()
    return {"accuracy": evaluator.eval_pred_list(pred_list)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, required=True,
                        help="Path or HF ID of the base model (e.g. Efficient-Large-Model/NVILA-8B)")
    parser.add_argument("--stage2-ckpt", type=str, required=True,
                        help="Path to stage2 checkpoint dir (contains adapter_model.safetensors "
                             "and global_step*/ with DeepSpeed ZeRO shards)")
    parser.add_argument("--conv-mode", type=str, required=True)
    parser.add_argument("--max-tiles", type=int, default=12)
    parser.add_argument("--generation-config", type=json.loads)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--no-cache",
        dest="use_cache",
        action="store_false",
        default=False,
        help="Disable LinearGatedCache (full-sequence mode, matches training). This is the default.",
    )
    parser.add_argument(
        "--use-cache",
        dest="use_cache",
        action="store_true",
        help="Enable LinearGatedCache (fast recurrent inference).",
    )
    args = parser.parse_args()

    data_path = DATASETS["textvqa"]["data_path"]
    image_dir = DATASETS["textvqa"]["image_dir"]
    answer_path = DATASETS["textvqa"]["answer_path"]

    dist.init()
    devices = range(dist.local_rank(), torch.cuda.device_count(), dist.local_size())
    torch.cuda.set_device(devices[0])

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_mode].copy()

    model = load_linear_attn_stage2_alone(
        base_model_path=args.base_model,
        stage2_ckpt_path=args.stage2_ckpt,
        devices=devices,
    )

    model.config.min_tiles = 1
    model.config.max_tiles = args.max_tiles
    model.llm.config.min_tiles = 1
    model.llm.config.max_tiles = args.max_tiles

    context_length = model.tokenizer.model_max_length
    if args.max_tiles > 12:
        context_length = max(context_length, int(args.max_tiles / 12.0 * 4096))
    model.config.model_max_length = context_length
    model.config.tokenizer_model_max_length = context_length
    model.llm.config.model_max_length = context_length
    model.llm.config.tokenizer_model_max_length = context_length
    model.tokenizer.model_max_length = context_length

    generation_config = model.default_generation_config
    if args.generation_config is not None:
        generation_config.update(**args.generation_config)
    generation_config.use_cache = args.use_cache
    logger.info(f"use_cache={generation_config.use_cache} "
                f"({'recurrent/cache' if args.use_cache else 'full-sequence/no-cache'})")

    instances = io.load(data_path)[dist.rank() :: dist.size()]

    outputs = []
    for instance in tqdm(instances, disable=not dist.is_main()):
        image = Image.open(os.path.join(image_dir, instance["image"]))
        question = instance["text"]
        response = model.generate_content([image, question], generation_config=generation_config)
        outputs.append({"question_id": instance["question_id"], "prompt": question, "text": response})

    if dist.size() > 1:
        outputs = dist.gather(outputs, dst=0)
        if not dist.is_main():
            return
        outputs = list(itertools.chain(*outputs))

    os.makedirs(args.output_dir, exist_ok=True)
    io.save(os.path.join(args.output_dir, "outputs.jsonl"), outputs)

    answers = io.load(answer_path)
    metrics = eval_single(outputs, answers)
    io.save(os.path.join(args.output_dir, "metrics.json"), metrics)
    logger.info(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
