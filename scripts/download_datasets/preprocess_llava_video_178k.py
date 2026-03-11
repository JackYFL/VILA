#!/usr/bin/env python3
"""
预处理 LLaVA-Video-178K 数据集
1. 批量解压所有 *.tar.gz 视频压缩包（并发，支持断点跳过）
2. 合并所有子集的 *_processed.json 为一个统一的训练 JSONL 文件

用法：
    # 只解压视频
    python preprocess_llava_video_178k.py --input_dir ./LLaVA-Video-178K --output_dir ./LLaVA-Video-178K-processed

    # 只合并 JSON 标注
    python preprocess_llava_video_178k.py --input_dir ./LLaVA-Video-178K --output_dir ./LLaVA-Video-178K-processed --skip_extract

    # 指定并发数
    python preprocess_llava_video_178k.py --input_dir ./LLaVA-Video-178K --output_dir ./LLaVA-Video-178K-processed --num_workers 16
"""

import argparse
import json
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="预处理 LLaVA-Video-178K 数据集")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./LLaVA-Video-178K",
        help="下载好的数据集根目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./LLaVA-Video-178K-processed",
        help="处理后的输出目录",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="并发解压线程数 (默认: 8)",
    )
    parser.add_argument(
        "--skip_extract",
        action="store_true",
        help="跳过解压步骤，只合并 JSON 标注文件",
    )
    parser.add_argument(
        "--skip_merge",
        action="store_true",
        help="跳过合并 JSON 步骤，只解压视频",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────
# Step 1: 解压视频
# ──────────────────────────────────────────────

def find_all_tarballs(input_dir):
    """递归找出所有 *.tar.gz 文件"""
    tarballs = sorted(Path(input_dir).rglob("*.tar.gz"))
    return tarballs


def is_extracted(tarball_path, video_dir):
    """判断该压缩包是否已经解压过（用 .done 标记文件）"""
    done_flag = os.path.join(video_dir, ".done", tarball_path.name + ".done")
    return os.path.exists(done_flag)


def mark_extracted(tarball_path, video_dir):
    done_dir = os.path.join(video_dir, ".done")
    os.makedirs(done_dir, exist_ok=True)
    open(os.path.join(done_dir, tarball_path.name + ".done"), "w").close()


def extract_tarball(args):
    tarball_path, video_dir = args
    if is_extracted(tarball_path, video_dir):
        return str(tarball_path), True, "already extracted"
    try:
        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(path=video_dir)
        mark_extracted(tarball_path, video_dir)
        return str(tarball_path), True, "ok"
    except Exception as e:
        return str(tarball_path), False, str(e)


def extract_all(input_dir, output_dir, num_workers):
    print("=" * 60)
    print("Step 1: 解压视频压缩包")
    print("=" * 60)

    video_dir = os.path.join(output_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    tarballs = find_all_tarballs(input_dir)
    if not tarballs:
        print("未找到任何 *.tar.gz 文件，请检查 --input_dir 路径")
        return

    already = sum(1 for t in tarballs if is_extracted(t, video_dir))
    print(f"共找到 {len(tarballs)} 个压缩包，已解压 {already} 个，待解压 {len(tarballs) - already} 个")
    print(f"视频输出目录: {video_dir}")
    print(f"并发线程数: {num_workers}")

    task_args = [(t, video_dir) for t in tarballs]
    failed = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(extract_tarball, arg): arg[0] for arg in task_args}
        with tqdm(total=len(futures), desc="解压进度") as pbar:
            for future in as_completed(futures):
                path, success, msg = future.result()
                if not success:
                    failed.append((path, msg))
                pbar.update(1)

    print(f"\n解压完成: 成功 {len(tarballs) - len(failed)} / {len(tarballs)}")
    if failed:
        failed_log = os.path.join(output_dir, "failed_extract.txt")
        with open(failed_log, "w") as f:
            for p, reason in failed:
                f.write(f"{p}\t{reason}\n")
        print(f"失败 {len(failed)} 个，详见: {failed_log}")


# ──────────────────────────────────────────────
# Step 2: 合并 JSON 标注
# ──────────────────────────────────────────────

SKIP_DIRS = {".cache", "gpt4o_caption_prompt", "gpt4o_qa_prompt"}


def find_all_json(input_dir):
    """找出所有子集中的 *_processed.json 文件"""
    json_files = []
    for subset_dir in sorted(Path(input_dir).iterdir()):
        if not subset_dir.is_dir():
            continue
        if subset_dir.name.startswith(".") or subset_dir.name in SKIP_DIRS:
            continue
        for jf in sorted(subset_dir.glob("*_processed.json")):
            json_files.append(jf)
    return json_files


def merge_annotations(input_dir, output_dir):
    print("=" * 60)
    print("Step 2: 合并标注 JSON 文件")
    print("=" * 60)

    json_files = find_all_json(input_dir)
    if not json_files:
        print("未找到任何 *_processed.json 文件")
        return

    print(f"共找到 {len(json_files)} 个标注文件")

    all_data = []
    for jf in tqdm(json_files, desc="加载标注"):
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            all_data.extend(data)
        else:
            all_data.append(data)

    # 重新分配全局唯一 id
    for i, item in enumerate(all_data):
        item["id"] = i

    output_file = os.path.join(output_dir, "llava_video_178k_train.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in all_data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"\n合并完成: 共 {len(all_data)} 条样本")
    print(f"输出文件: {output_file}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not args.skip_extract:
        extract_all(input_dir, output_dir, args.num_workers)

    if not args.skip_merge:
        merge_annotations(input_dir, output_dir)

    print("=" * 60)
    print(f"预处理完成！输出目录: {output_dir}")
    print("目录结构：")
    print(f"  {output_dir}/")
    print(f"    videos/                          # 解压后的视频（保留原始相对路径）")
    print(f"    llava_video_178k_train.jsonl     # 合并后的训练标注文件")
    print(f"    failed_extract.txt               # 解压失败列表（如有）")


if __name__ == "__main__":
    main()
