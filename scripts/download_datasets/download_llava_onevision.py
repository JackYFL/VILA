#!/usr/bin/env python3
"""
下载 LLaVA-OneVision 1.6M 数据集 (lmms-lab/LLaVA-OneVision-Data)
支持断点续传，可指定下载目录和并发数
"""

import argparse
import os

from huggingface_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(description="下载 LLaVA-OneVision 1.6M 数据集")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./LLaVA-OneVision-Data",
        help="数据集保存路径 (默认: ./LLaVA-OneVision-Data)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="并发下载线程数 (默认: 8)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace access token (如数据集需要登录)",
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="+",
        default=None,
        help="只下载指定子集，例如 --include 'ai2d*' 'chartqa*'",
    )
    parser.add_argument(
        "--ignore_patterns",
        type=str,
        nargs="+",
        default=None,
        help="排除匹配的文件，例如 --ignore_patterns '*.parquet'",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    repo_id = "lmms-lab/LLaVA-OneVision-Data"
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"数据集: {repo_id}")
    print(f"保存路径: {output_dir}")
    print(f"并发线程数: {args.num_workers}")
    if args.include:
        print(f"只下载: {args.include}")
    if args.ignore_patterns:
        print(f"排除文件: {args.ignore_patterns}")
    print("-" * 60)

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        token=args.token,
        allow_patterns=args.include,
        ignore_patterns=args.ignore_patterns,
        max_workers=args.num_workers,
        resume_download=True,
    )

    print("-" * 60)
    print(f"下载完成！数据保存在: {output_dir}")
    print()
    print("后续处理步骤：")
    print("  1. 将 Parquet 转换为 JSONL 格式：")
    print("     python VILA/data_prepare/sft/preprocess_llava_onevision.py \\")
    print(f"       --dataset_path {output_dir} \\")
    print("       --save_path ./LLaVA-OneVision-Data-processed/")
    print()
    print("  2. 合并各子集为统一训练文件：")
    print("     python VILA/data_prepare/sft/merge_llava_onevision.py \\")
    print("       --save_path ./LLaVA-OneVision-Data-processed/")


if __name__ == "__main__":
    main()
