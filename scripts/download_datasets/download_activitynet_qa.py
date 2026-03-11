#!/usr/bin/env python3
"""
下载 ActivityNet-QA 数据集
- QA 标注：从 GitHub (MILVLG/activitynet-qa) 下载
- 视频：从 YouTube 通过 yt-dlp 下载（需要提前安装 yt-dlp）

依赖：
    pip install requests yt-dlp tqdm
"""

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

# ActivityNet-QA 标注文件（来自 GitHub: MILVLG/activitynet-qa）
ANNOTATION_FILES = {
    "train_q.json": "https://github.com/MILVLG/activitynet-qa/raw/master/dataset/train_q.json",
    "train_a.json": "https://github.com/MILVLG/activitynet-qa/raw/master/dataset/train_a.json",
    "val_q.json":   "https://github.com/MILVLG/activitynet-qa/raw/master/dataset/val_q.json",
    "val_a.json":   "https://github.com/MILVLG/activitynet-qa/raw/master/dataset/val_a.json",
    "test_q.json":  "https://github.com/MILVLG/activitynet-qa/raw/master/dataset/test_q.json",
    "test_a.json":  "https://github.com/MILVLG/activitynet-qa/raw/master/dataset/test_a.json",
}


def parse_args():
    parser = argparse.ArgumentParser(description="下载 ActivityNet-QA 数据集")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ActivityNet-QA",
        help="数据集保存路径 (默认: ./ActivityNet-QA)",
    )
    parser.add_argument(
        "--download_videos",
        action="store_true",
        help="同时下载视频（需要 yt-dlp，视频来自 YouTube 可能受地区限制）",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="并发下载线程数 (默认: 8)",
    )
    parser.add_argument(
        "--video_resolution",
        type=str,
        default="360",
        choices=["360", "480", "720", "best"],
        help="视频分辨率 (默认: 360)",
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="代理地址，例如 socks5://127.0.0.1:1080（YouTube 下载可能需要）",
    )
    parser.add_argument(
        "--cookies_from_browser",
        type=str,
        default=None,
        metavar="BROWSER",
        help="从浏览器读取 cookies 以绕过 YouTube 机器人验证，例如 chrome、firefox、edge",
    )
    parser.add_argument(
        "--cookies",
        type=str,
        default=None,
        metavar="FILE",
        help="指定 cookies 文件路径（Netscape 格式，可用 yt-dlp --cookies-from-browser 导出）",
    )
    return parser.parse_args()


def download_annotations(output_dir, proxy=None):
    """从 GitHub 下载 QA 标注文件"""
    print("=" * 60)
    print("Step 1: 下载 QA 标注文件 (来自 GitHub: MILVLG/activitynet-qa)")
    print("=" * 60)
    annotation_dir = os.path.join(output_dir, "annotations")
    os.makedirs(annotation_dir, exist_ok=True)

    # 显式设置代理：避免环境变量（如 ALL_PROXY/HTTPS_PROXY）中无效的代理干扰
    if proxy:
        proxies = {"http": proxy, "https": proxy}
    else:
        proxies = {"http": None, "https": None}

    session = requests.Session()
    session.proxies.update(proxies)

    for filename, url in ANNOTATION_FILES.items():
        save_path = os.path.join(annotation_dir, filename)
        if os.path.exists(save_path):
            print(f"  已存在，跳过: {filename}")
            continue
        print(f"  下载: {filename} ...")
        try:
            resp = session.get(url, timeout=60)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            print(f"\n错误：连接 GitHub 失败: {e}")
            print("请检查网络连接，或使用 --proxy 参数指定可用的代理，例如：")
            print("  python download_activitynet_qa.py --proxy socks5://127.0.0.1:1080")
            raise SystemExit(1)
        with open(save_path, "wb") as f:
            f.write(resp.content)
        print(f"  完成: {filename}")

    print(f"标注文件已保存至: {annotation_dir}\n")
    return annotation_dir


def collect_video_ids(annotation_dir):
    """从 *_q.json 标注文件中收集所有视频 ID"""
    video_ids = set()
    for split in ["train", "val", "test"]:
        json_path = os.path.join(annotation_dir, f"{split}_q.json")
        if not os.path.exists(json_path):
            continue
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        # 格式: [{"question_id": ..., "video_name": "v_XXXXX", "question": ...}, ...]
        for item in data:
            vid = item.get("video_name", "")
            if vid:
                video_ids.add(vid.replace("v_", ""))
    return sorted(video_ids)


def download_single_video(args):
    """用 yt-dlp 下载单个视频"""
    video_id, video_dir, resolution, proxy, cookies_from_browser, cookies = args
    output_path = os.path.join(video_dir, f"v_{video_id}.mp4")
    if os.path.exists(output_path):
        return video_id, True, "already exists"

    url = f"https://www.youtube.com/watch?v={video_id}"

    if resolution == "best":
        fmt = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
    else:
        fmt = f"bestvideo[height<={resolution}][ext=mp4]+bestaudio[ext=m4a]/best[height<={resolution}][ext=mp4]/best"

    cmd = [
        "yt-dlp",
        "-f", fmt,
        "-o", output_path,
        "--quiet",
        "--no-warnings",
        "--merge-output-format", "mp4",
        url,
    ]
    if proxy:
        cmd += ["--proxy", proxy]
    if cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]
    if cookies:
        cmd += ["--cookies", cookies]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        return video_id, True, "ok"
    except subprocess.CalledProcessError as e:
        return video_id, False, e.stderr.decode("utf-8", errors="ignore").strip()
    except subprocess.TimeoutExpired:
        return video_id, False, "timeout"


def download_videos(annotation_dir, output_dir, num_workers, resolution, proxy, cookies_from_browser, cookies):
    """并发下载所有视频"""
    print("=" * 60)
    print("Step 2: 下载视频 (来自 YouTube，通过 yt-dlp)")
    print("=" * 60)

    try:
        subprocess.run(["yt-dlp", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("错误：未找到 yt-dlp，请先安装：pip install yt-dlp")
        return

    video_ids = collect_video_ids(annotation_dir)
    if not video_ids:
        print("警告：未从标注文件中找到视频 ID，请检查标注文件是否下载成功")
        return

    print(f"共找到 {len(video_ids)} 个视频，分辨率: {resolution}p，并发数: {num_workers}")
    if proxy:
        print(f"使用代理: {proxy}")
    if cookies_from_browser:
        print(f"使用浏览器 cookies: {cookies_from_browser}")
    if cookies:
        print(f"使用 cookies 文件: {cookies}")

    video_dir = os.path.join(output_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    failed_ids = []
    task_args = [(vid, video_dir, resolution, proxy, cookies_from_browser, cookies) for vid in video_ids]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(download_single_video, arg): arg[0] for arg in task_args}
        with tqdm(total=len(futures), desc="下载视频") as pbar:
            for future in as_completed(futures):
                video_id, success, msg = future.result()
                if not success and msg != "already exists":
                    failed_ids.append((video_id, msg))
                pbar.update(1)

    success_count = len(video_ids) - len(failed_ids)
    print(f"\n完成: 成功 {success_count} / {len(video_ids)}")
    if failed_ids:
        failed_log = os.path.join(output_dir, "failed_videos.txt")
        with open(failed_log, "w") as f:
            for vid, reason in failed_ids:
                f.write(f"{vid}\t{reason}\n")
        print(f"失败 {len(failed_ids)} 个，详见: {failed_log}")
        print("提示：可重新运行脚本，已下载的视频会自动跳过")


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    annotation_dir = download_annotations(output_dir, proxy=args.proxy)

    if args.download_videos:
        download_videos(annotation_dir, output_dir, args.num_workers, args.video_resolution, args.proxy, args.cookies_from_browser, args.cookies)
    else:
        print("提示：仅下载了 QA 标注文件。若需下载视频，请添加 --download_videos 参数。")

    print("=" * 60)
    print(f"ActivityNet-QA 下载完成！保存路径: {output_dir}")
    print("目录结构：")
    print(f"  {output_dir}/")
    print(f"    annotations/      # train_q.json, train_a.json, val_q.json ...")
    if args.download_videos:
        print(f"    videos/           # v_<video_id>.mp4")
        print(f"    failed_videos.txt # 下载失败的视频（如有）")


if __name__ == "__main__":
    main()
