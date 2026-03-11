#!/usr/bin/env python3
"""
下载 LLaVA-Video-178K 数据集 (lmms-lab/LLaVA-Video-178K)
支持断点续传，可指定下载目录和并发数
"""

import argparse
import os

from huggingface_hub import snapshot_download

# LLaVA-Video-178K 包含的子集
SUBSETS = [
    "activitynet_qa",
    "ego4d",
    "kinetics_400_700",
    "kinetics_600",
    "nextqa",
    "perceptiontest",
    "shareVideoGPTV",
    "videochatgpt",
    "youcook2",
]


def parse_args():
    parser = argparse.ArgumentParser(description="下载 LLaVA-Video-178K 数据集")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./LLaVA-Video-178K",
        help="数据集保存路径 (默认: ./LLaVA-Video-178K)",
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
        "--subsets",
        type=str,
        nargs="+",
        default=None,
        choices=SUBSETS,
        help=(
            "只下载指定子集（默认下载全部）。可选: " + ", ".join(SUBSETS)
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    repo_id = "lmms-lab/LLaVA-Video-178K"
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 根据子集构建 allow_patterns
    if args.subsets:
        allow_patterns = [f"{subset}/*" for subset in args.subsets] + ["*.json", "README.md"]
    else:
        allow_patterns = None

    print(f"数据集: {repo_id}")
    print(f"保存路径: {output_dir}")
    print(f"并发线程数: {args.num_workers}")
    if args.subsets:
        print(f"下载子集: {args.subsets}")
    else:
        print(f"下载子集: 全部 ({', '.join(SUBSETS)})")
    print("-" * 60)

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        token=args.token,
        allow_patterns=allow_patterns,
        max_workers=args.num_workers,
        resume_download=True,
    )

    print("-" * 60)
    print(f"下载完成！数据保存在: {output_dir}")
    print()
    print("数据集结构：")
    print(f"  {output_dir}/")
    for subset in (args.subsets or SUBSETS):
        print(f"    {subset}/")
        print(f"      *.mp4          # 视频文件")
        print(f"      *.json         # 标注文件")


if __name__ == "__main__":
    main()
