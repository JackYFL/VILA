# NVILA-8B SFT 训练指南

本文档介绍如何使用以下两个脚本对 NVILA-8B 进行监督微调（SFT），并梳理 VILA 项目的整体结构与训练流程。

| 脚本 | 数据集 | 模态 |
|------|--------|------|
| `train_nvila_8b_llava_onevision_img.sh` | LLaVA-OneVision-Data-processed（89 个子集） | 图像 |
| `train_nvila_8b_llava_onevision.sh` | LLaVA-Video-178K-processed（约 165 万条） | 视频 |

---

## 目录

1. [快速开始](#快速开始)
2. [环境准备](#环境准备)
3. [数据集准备](#数据集准备)
4. [训练脚本详解](#训练脚本详解)
   - [图像 SFT 脚本](#图像-sft-脚本-train_nvila_8b_llava_onevision_imgsh)
   - [视频 SFT 脚本](#视频-sft-脚本-train_nvila_8b_llava_ovisionsh)
   - [两脚本关键差异对比](#两脚本关键差异对比)
5. [VILA 项目概览](#vila-项目概览)
6. [训练流程总览](#训练流程总览)
7. [关键超参数说明](#关键超参数说明)
8. [输出与监控](#输出与监控)
9. [常见问题](#常见问题)

---

## 快速开始

```bash
# 必须从 VILA 根目录运行
cd /mnt/localssd/VILA

# 图像 SFT（LLaVA-OneVision-Data-processed）
bash scripts/train_nvila_8b_llava_onevision_img.sh

# 视频 SFT（LLaVA-Video-178K-processed）
bash scripts/train_nvila_8b_llava_onevision.sh
```

覆盖默认参数示例：

```bash
# 使用本地 checkpoint
MODEL_PATH=/path/to/local/nvila-8b bash scripts/train_nvila_8b_llava_onevision_img.sh

# 自定义运行名称（影响输出目录）
RUN_NAME=my-img-experiment bash scripts/train_nvila_8b_llava_onevision_img.sh

# 调整 GPU 数量（单节点 4 卡）
GPUS_PER_NODE=4 bash scripts/train_nvila_8b_llava_onevision_img.sh
```

---

## 环境准备

### 安装依赖

```bash
# 推荐使用 conda 环境
bash environment_setup.sh

# 或手动安装
pip install -e ".[train]"
```

### 主要依赖版本

| 包 | 版本 |
|----|------|
| PyTorch | 2.3.0 |
| Transformers | 4.46.0 |
| DeepSpeed | 0.9.5 |
| Accelerate | 0.34.2 |
| decord2 | 2.0.0 |
| wandb | 最新 |

---

## 数据集准备

### LLaVA-OneVision-Data-processed（图像脚本使用）

```
/mnt/localssd/datasets/LLaVA-OneVision-Data-processed/
├── metadata/                      # 89 个 JSONL 元数据文件
│   ├── ai2d(cauldron,llava_format)_train.jsonl
│   ├── CLEVR-Math(MathV360K)_train.jsonl
│   ├── sharegpt4v(coco)_train.jsonl
│   └── ...（共 89 个子集）
└── images/                        # 图像文件根目录
    ├── ai2d(cauldron,llava_format)/
    ├── CLEVR-Math(MathV360K)/
    └── ...
```

**JSONL 格式：**

```json
{
  "id": 0,
  "image": "./LLaVA-OneVision-Data-processed/images/ai2d(cauldron,llava_format)/0.png",
  "conversations": [
    {"from": "human", "value": "<image>\n问题内容"},
    {"from": "gpt",   "value": "回答内容"}
  ]
}
```

> `image` 字段是相对于 `/mnt/localssd/datasets` 的路径（含 `./` 前缀），脚本将 `media_dir` 指向该目录。

### LLaVA-Video-178K-processed（视频脚本使用）

```
/mnt/localssd/datasets/LLaVA-Video-178K-processed/
├── llava_video_178k_train.jsonl   # 单一元数据文件（约 165 万条）
├── failed_extract.txt             # 预处理阶段解码失败的视频列表
└── videos/                        # 视频文件根目录
    ├── ActivityNet-QA/
    ├── NextQA/
    ├── academic_source/
    │   └── Charades/
    ├── liwei_youtube_videos/
    └── perception_test/
```

**JSONL 格式：**

```json
{
  "id": 0,
  "data_source": "0_30_s_academic_v0_1",
  "video": "academic_source/Charades/RW587.mp4",
  "conversations": [
    {"from": "human", "value": "<video>\n问题内容"},
    {"from": "gpt",   "value": "回答内容"}
  ]
}
```

> `video` 字段是相对于 `videos/` 目录的路径，脚本将 `media_dir` 指向该目录。

---

## 训练脚本详解

两个脚本共享相同的四步执行结构：

```
脚本结构
│
├── [1] 参数配置        ← 可通过环境变量覆盖的默认值
├── [2] 数据集注册      ← 动态生成临时 YAML，注册到 VILA 数据系统
├── [3] 分布式训练配置  ← source scripts/setups/train.sh
└── [4] 启动训练        ← torchrun + llava/train/train_mem.py
```

### VILA 数据注册机制

VILA 使用基于 YAML + Hydra 的数据注册系统，不直接在命令行传路径。脚本在运行时动态创建临时 YAML 并通过 `VILA_DATASETS` 环境变量注入：

```bash
export VILA_DATASETS="default,/tmp/my_datasets_xxx.yaml"
```

训练进程启动时，`llava/data/builder.py` 从该 YAML 加载数据集定义：

```python
DATASETS = register_datasets()  # 在 import 时执行，读取 VILA_DATASETS
```

每个数据集条目通过 `_target_` 指定类，由 Hydra 的 `instantiate()` 实例化，支持 `LLaVADataset`、`LLaVANextDataset` 等。训练结束后临时文件自动删除（`trap "rm -f ..." EXIT`）。

---

### 图像 SFT 脚本（`train_nvila_8b_llava_onevision_img.sh`）

#### 参数配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `Efficient-Large-Model/NVILA-8B` | 起始 checkpoint |
| `RUN_NAME` | `nvila-8b-llava-onevision-img-sft` | 运行名称 / 输出目录 |
| `GLOBAL_TRAIN_BATCH_SIZE` | `256` | 全局 batch size |
| `GRADIENT_ACCUMULATION_STEPS` | `4` | 梯度累积步数 |

**每卡 batch size：** `256 / 1 / 8 / 4 = 8`

#### 数据集注册方式

89 个 JSONL 文件逐一注册，动态生成的 YAML 片段示例：

```yaml
---
'ai2d(cauldron,llava_format)_train':
    _target_: llava.data.LLaVADataset
    data_path: /mnt/localssd/datasets/LLaVA-OneVision-Data-processed/metadata/ai2d(cauldron,llava_format)_train.jsonl
    media_dir: /mnt/localssd/datasets
'CLEVR-Math(MathV360K)_train':
    _target_: llava.data.LLaVADataset
    data_path: /mnt/localssd/datasets/LLaVA-OneVision-Data-processed/metadata/CLEVR-Math(MathV360K)_train.jsonl
    media_dir: /mnt/localssd/datasets
# ... 共 89 条
```

所有数据集名称排序后用 `+` 连接传给 `--data_mixture`：

```bash
DATA_MIXTURE="CLEVR-Math(MathV360K)_train+Evol-Instruct-GPT4-Turbo_train+..."
```

#### 核心训练参数

| 参数 | 值 | 说明 |
|------|----|------|
| `--vision_tower` | `paligemma-siglip-so400m-patch14-448` | SigLIP 视觉编码器 |
| `--mm_projector` | `mlp_downsample` | 3×3 下采样 MLP 投影层 |
| `--image_aspect_ratio` | `dynamic_s2` | 启用 S2 动态分辨率（9-tile） |
| `--dynamic_s2` | `True` | 开启多尺度图像编码 |
| `--s2_scales` | `"448,896,1344"` | 三级分辨率梯度 |
| `--model_max_length` | `8192` | 上下文长度 |
| `--learning_rate` | `1.5e-5` | 图像 SFT 学习率 |

---

### 视频 SFT 脚本（`train_nvila_8b_llava_onevision.sh`）

#### 参数配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `Efficient-Large-Model/NVILA-8B` | 起始 checkpoint |
| `RUN_NAME` | `nvila-8b-llava-video-178k-sft` | 运行名称 / 输出目录 |
| `GLOBAL_TRAIN_BATCH_SIZE` | `256` | 全局 batch size |
| `GRADIENT_ACCUMULATION_STEPS` | `8` | 梯度累积步数（视频更大） |

**每卡 batch size：** `256 / 1 / 8 / 8 = 4`

#### 数据集注册方式

单文件注册，`is_video: true` 显式标记为视频数据集（`LLaVADataset` 也可通过检测 `"video"` 字段自动识别）：

```yaml
---
llava-video-178k:
    _target_: llava.data.LLaVADataset
    data_path: /mnt/localssd/datasets/LLaVA-Video-178K-processed/llava_video_178k_train.jsonl
    media_dir: /mnt/localssd/datasets/LLaVA-Video-178K-processed/videos
    is_video: true
```

#### 核心训练参数

| 参数 | 值 | 说明 |
|------|----|------|
| `--vision_tower` | `paligemma-siglip-so400m-patch14-448` | SigLIP 视觉编码器 |
| `--mm_projector` | `mlp_downsample_2x2_fix` | 2×2 下采样 MLP 投影层 |
| `--image_aspect_ratio` | `resize` | 视频帧直接 resize，不分 tile |
| `--video_encoder` | `TSPVideoEncoder` | 时序空间金字塔视频编码器 |
| `--num_video_frames` | `256` | 每段视频采样帧数 |
| `--num_time_tokens` | `100` | 时间位置编码 token 数量 |
| `--model_max_length` | `16384` | 视频需要更长上下文 |
| `--learning_rate` | `2e-5` | 视频 SFT 学习率 |

> **TSPVideoEncoder**（Temporal Spatial Pyramid）：通过 `pool_sizes: [[8,1,1]]` 在时间维度做 8× 池化，将 256 帧压缩为 32 个时间步，有效降低序列长度。

---

### 两脚本关键差异对比

| 维度 | 图像脚本 | 视频脚本 |
|------|----------|----------|
| 数据集文件 | 89 个 JSONL | 1 个 JSONL |
| 数据量 | 多个子集混合 | ~165 万条视频 QA |
| `media_dir` | `/mnt/localssd/datasets` | `.../LLaVA-Video-178K-processed/videos` |
| `image_aspect_ratio` | `dynamic_s2`（9-tile） | `resize` |
| `mm_projector` | `mlp_downsample` | `mlp_downsample_2x2_fix` |
| `dynamic_s2` | `True` | 无此参数 |
| 视频编码器 | 无 | `TSPVideoEncoder` |
| `num_video_frames` | 无 | `256` |
| `model_max_length` | `8192` | `16384` |
| `learning_rate` | `1.5e-5` | `2e-5` |
| `gradient_accumulation` | `4` | `8` |
| 每卡 batch size | `8` | `4` |

---

## VILA 项目概览

### 项目定位

VILA（**V**isual **I**nstruction-tuned **L**anguage model with **A**lignment）是 NVIDIA 开源的多模态大语言模型系列，专注于高效的视频和多图理解。NVILA（VILA 2.0）在此基础上引入全栈效率优化：更快的推理速度、更低的训练成本、更强的性能。

### 目录结构

```
VILA/
├── llava/                       # 核心实现
│   ├── model/                   # 模型架构
│   │   ├── llava_arch.py        # VLM 主体架构
│   │   ├── multimodal_encoder/  # 视觉编码器（SigLIP、CLIP、PS3 等）
│   │   ├── multimodal_projector/# 视觉-语言投影层
│   │   ├── encoders/            # 视频编码器（TSPVideoEncoder 等）
│   │   └── language_model/      # 语言模型组件
│   ├── train/                   # 训练入口
│   │   ├── train_mem.py         # 主训练脚本（节省显存版）
│   │   └── args.py              # 所有训练参数定义
│   ├── data/                    # 数据加载系统
│   │   ├── dataset_impl/        # 具体数据集实现（LLaVADataset 等）
│   │   ├── builder.py           # 数据集注册与构建
│   │   └── registry/            # YAML 注册表
│   │       ├── datasets/        # 数据集定义（default.yaml）
│   │       └── mixtures.yaml    # 预定义数据混合方案
│   └── eval/                    # 评测框架
├── scripts/                     # 训练与评测脚本
│   ├── NVILA/                   # NVILA 完整 4 阶段训练流程
│   ├── NVILA-Lite/              # NVILA-Lite 精简训练流程
│   ├── eval/                    # 评测脚本（视频/图像 benchmark）
│   ├── setups/train.sh          # 分布式训练公共配置
│   ├── zero3.json               # DeepSpeed ZeRO-3 配置
│   ├── zero3_gradient_clipping.json
│   ├── train_nvila_8b_llava_onevision_img.sh  ← 图像 SFT
│   └── train_nvila_8b_llava_onevision.sh      ← 视频 SFT
├── longvila/                    # 长视频理解变体
├── vila_hd/                     # 高分辨率变体（PS3 编码器）
├── finetuning/                  # 微调工具
├── serving/                     # 生产部署
└── demo_trt_llm/                # TensorRT-LLM 部署示例
```

### 模型系列

| 模型 | 参数量 | 特点 |
|------|--------|------|
| NVILA-3B | ~3B | 轻量，适合边缘部署 |
| NVILA-8B | ~8B | 均衡，本脚本使用 |
| NVILA-15B | ~15B | 高精度 |
| NVILA-Lite-8B | ~8B | 简化训练流程，推理速度稍快 |
| NVILA-Video-8B | ~8B | 视频专项优化版本 |

---

## 训练流程总览

NVILA-8B 的完整训练分为 4 个阶段，本文档两个脚本均属于基于公开 NVILA-8B 的继续微调（续训阶段 3 或阶段 4）：

```
阶段 1: 视觉对齐（scripts/NVILA/stage1_9tile.sh）
  输入: Qwen2.5-7B-Instruct（纯语言模型）
  目标: 训练 MM Projector，建立图像-文本初步对齐
  学习率: 1e-3  |  只训练 projector
        │
        ▼
阶段 1.5: 视觉塔微调（scripts/NVILA/stage15_9tile.sh）
  输入: 阶段 1 输出
  目标: 联合微调 Vision Tower + Projector，提升图像特征质量
  学习率: 2e-5  |  训练 vision tower + projector
        │
        ▼
阶段 2: 语言模型预训练（scripts/NVILA/stage2_9tile.sh）
  输入: 阶段 1.5 输出
  目标: 让语言模型学会理解多模态输入
  学习率: 2e-5  |  训练 projector + language model
        │
        ▼
阶段 3: 图像 SFT（scripts/NVILA/stage3_9tile.sh）
  输入: 阶段 2 输出
  目标: 全量指令微调，强化图像理解能力
  学习率: 1.5e-5  |  训练全部模块
        │
        ├─────────────────────────────────────────────┐
        ▼                                             ▼
  图像 SFT 续训                                  视频 SFT
  train_nvila_8b_llava_onevision_img.sh          train_nvila_8b_llava_onevision.sh
  输入: Efficient-Large-Model/NVILA-8B           输入: Efficient-Large-Model/NVILA-8B
  数据: LLaVA-OneVision（89 子集）               数据: LLaVA-Video-178K（165 万条）
  特点: 9-tile 动态分辨率，图像多任务            特点: 视频编码器，时序理解
```

### 动态分辨率（9-tile）说明

图像脚本使用 S2（Scale-Square）动态分辨率技术，将高分辨率图像切分为 3 种尺度（448/896/1344）的 tile 分别编码，最多支持 9 个 tile（1 个全图 + 最多 8 个局部裁剪），大幅提升文字识别、细节理解能力。视频脚本改为固定 resize，因为视频帧中的局部细节不是核心，时序理解才是重点。

---

## 关键超参数说明

### 为什么图像脚本 model_max_length=8192，视频脚本为 16384？

- **图像**：9-tile 模式下每张图最多产生约 1K 视觉 token，8K 上下文足够
- **视频**：256 帧经 2×2 池化后每帧约 256 token，总计 ~6.5K 视觉 token，加上文本需 16K

### 为什么使用 ZeRO-3 + gradient clipping（max_grad_norm=5.0）？

- **ZeRO-3**：将模型参数、梯度、优化器状态分片到所有 GPU，8B 模型在 8×A100 80GB 上可以正常训练
- **gradient clipping**：图像和视频全量 SFT 时梯度波动较大，裁剪防止训练发散

### 为什么视频脚本 gradient_accumulation=8 而图像为 4？

视频每条样本的 token 数远多于图像，相同全局 batch size 下视频每卡实际显存占用更高，增大累积步数可降低每步的显存峰值。

---

## 输出与监控

### 输出目录

```
runs/train/
├── nvila-8b-llava-onevision-img-sft/   # 图像脚本输出
│   └── model/
│       ├── checkpoint-100/
│       └── checkpoint-200/
└── nvila-8b-llava-video-178k-sft/      # 视频脚本输出
    └── model/
        ├── checkpoint-100/
        └── ...（每 100 步保存，只保留最新 1 个）
```

### W&B 监控

两个脚本均自动上报 W&B：
- **项目名**：`vila`
- **运行名**：即 `RUN_NAME`，支持断点续训（`WANDB_RESUME=allow`）

如不需要 W&B，将最后一行改为 `--report_to none`。

---

## 常见问题

**Q: 图像脚本报错 `Dataset 'xxx_train' is not found`**

确认 `VILA_DATASETS` 已正确导出，且临时 YAML 文件在训练进程启动前已创建。脚本内部已自动处理，通常不会出现此问题。若手动调试，可在 `torchrun` 前加 `echo $VILA_DATASETS` 验证。

**Q: 视频加载失败（`failed_extract.txt` 中有很多记录）**

`failed_extract.txt` 记录了预处理阶段解码失败的视频，训练时跳过这些样本是正常的，不影响训练。

**Q: 显存不足（OOM）**

图像脚本：
1. 减少 `--s2_scales`（如只用 `"448,896"`，降低 tile 数量）
2. 增大 `GRADIENT_ACCUMULATION_STEPS`，等比缩小 `GLOBAL_TRAIN_BATCH_SIZE`

视频脚本：
1. 减少 `--num_video_frames`（如改为 128）
2. 增大 `GRADIENT_ACCUMULATION_STEPS`，等比缩小 `GLOBAL_TRAIN_BATCH_SIZE`

**Q: 如何从已有的 checkpoint 继续训练？**

```bash
# 图像续训
MODEL_PATH=runs/train/nvila-8b-llava-onevision-img-sft/model/checkpoint-500 \
    bash scripts/train_nvila_8b_llava_onevision_img.sh

# 视频续训
MODEL_PATH=runs/train/nvila-8b-llava-video-178k-sft/model/checkpoint-500 \
    bash scripts/train_nvila_8b_llava_onevision.sh
```

**Q: 如何推理/评测训练后的模型？**

```bash
# 图像推理
vila-infer --model-path runs/train/nvila-8b-llava-onevision-img-sft/model \
    --prompt "描述这张图片" --image path/to/image.jpg

# 视频推理
vila-infer --model-path runs/train/nvila-8b-llava-video-178k-sft/model \
    --prompt "描述这个视频" --video path/to/video.mp4

# 评测（以 EgoSchema 为例）
bash scripts/eval/egoschema.sh runs/train/nvila-8b-llava-video-178k-sft/model
```
