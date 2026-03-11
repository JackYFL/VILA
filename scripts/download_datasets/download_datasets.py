import os
import sys
from huggingface_hub import snapshot_download, hf_hub_download

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/datasets"
os.makedirs(ROOT, exist_ok=True)
print(f"[root] {ROOT}")


def maybe_snapshot(repo_id, local_dir, **kwargs):
    local_dir = os.path.join(ROOT, local_dir)
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        print(f"[skip] {local_dir} already exists")
        return
    print(f"[download] {repo_id} -> {local_dir}")
    snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False, **kwargs)


def maybe_file(repo_id, filename, local_dir, **kwargs):
    local_dir = os.path.join(ROOT, local_dir)
    dest = os.path.join(local_dir, filename)
    if os.path.isfile(dest):
        print(f"[skip] {dest} already exists")
        return
    print(f"[download] {repo_id}/{filename} -> {dest}")
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir,
                    local_dir_use_symlinks=False, **kwargs)


# ==============================================================
# SFT Stage Datasets
# ==============================================================

# --- VFlan: FLAN (TextFLAN) ---
maybe_snapshot('Open-Orca/FLAN', 'FLAN', repo_type='dataset')

# --- VFlan: M3IT ---
maybe_snapshot('MMInstruction/M3IT', 'M3IT', repo_type='dataset')

# --- LLaVA-1.5 Instruction Data (LLaVA-Next mixture base) ---
maybe_file('liuhaotian/LLaVA-Instruct-150K', 'llava_v1_5_mix665k.json', 'llava_instruct', repo_type='dataset')

# --- WIT ---
maybe_file('mit-han-lab/vila-dataset', 'wit_processed_538k.json', 'WIT', repo_type='dataset')

# --- Sherlock ---
maybe_file('mit-han-lab/vila-dataset', 'sherlock_317k.json', 'sherlock', repo_type='dataset')

# --- ScienceQA (JSON annotation + image data) ---
maybe_file('mit-han-lab/vila-dataset', 'scienceqa_train_12k.json', 'scienceqa', repo_type='dataset')
maybe_snapshot('derek-thomas/ScienceQA', 'scienceqa/images', repo_type='dataset')

# --- Shot2story (JSON only; videos must be downloaded separately per DATA.md) ---
maybe_file('mit-han-lab/vila-dataset', 'shot2story_shotonly.json', 'shot2story', repo_type='dataset')
maybe_snapshot('mhan/shot2story-videos', 'shot2story-videos', repo_type='dataset')

# --- Youcook2 (JSON only; videos must be downloaded from youcook2.eecs.umich.edu) ---
maybe_file('mit-han-lab/vila-dataset', 'youcook_filtered_v3.json', 'youcook2', repo_type='dataset')

# --- Vatex (JSON only; videos must be downloaded from eric-xw.github.io/vatex-website) ---
maybe_file('mit-han-lab/vila-dataset', 'vatex_filtered_v3.json', 'vatex', repo_type='dataset')

# --- ShareGPT_Video ---
maybe_snapshot('ShareGPTVideo/train_video_and_instruction', 'ShareGPT_Video', repo_type='dataset')

# ==============================================================
# NOTE: The following SFT datasets require manual steps:
# - LLaVA-Next mixture: follow https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat#prepare-training-datasets
# - GSM8K-ScRel-SFT: download train_use.jsonl from https://github.com/OFA-Sys/gsm8k-ScRel/blob/main/data/train_use.jsonl
# - Video_ChatGPT: follow https://github.com/mbzuai-oryx/Video-ChatGPT#video-instruction-dataset
# ==============================================================
