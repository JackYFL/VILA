#!/usr/bin/env bash
set -e

CONDA_ENV=${1:-""}
if [ -n "$CONDA_ENV" ]; then
    # This is required to activate conda environment
    eval "$(conda shell.bash hook)"

    conda create -n $CONDA_ENV python=3.10.14 -y
    conda activate $CONDA_ENV
    # This is optional if you prefer to use built-in nvcc
    conda install -c nvidia cuda-toolkit -y
else
    echo "Skipping conda environment creation. Make sure you have the correct environment activated."
fi

# This is required to enable PEP 660 support
pip install --upgrade pip setuptools

# Install FlashAttention2
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install VILA
pip install -e ".[train,eval]"

# Quantization requires the newest triton version, and introduce dependency issue
pip install triton==3.1.0

# numpy introduce a lot dependencies issues, separate from pyproject.yaml
# pip install numpy==1.26.4

# Replace transformers and deepspeed files
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv ./llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/

# Downgrade protobuf to 3.20 for backward compatibility
pip install protobuf==3.20.*

# Upgrade PyTorch to 2.5.1+cu124 (required for flash-linear-attention 0.4.2 compatibility)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install flash-linear-attention
pip install flash-linear-attention==0.4.2

# Upgrade wandb to fix protobuf import errors caused by the upgrade above
pip install --upgrade wandb

# Patch flash-linear-attention for Triton 3.1 compatibility
# Fix 1: @torch.compile incorrectly applied to torch.autograd.Function subclasses
python - <<'EOF'
import re, pathlib, site
sp = site.getsitepackages()[0]
for fname in ['fla/ops/attn/parallel.py', 'fla/ops/nsa/parallel.py']:
    p = pathlib.Path(sp) / fname
    if not p.exists():
        continue
    txt = p.read_text()
    new_txt = re.sub(r'@torch\.compile\n(class \w+\(torch\.autograd\.Function\))', r'\1', txt)
    if new_txt != txt:
        p.write_text(new_txt)
        print(f'Patched: {fname}')
EOF

# Fix 2: triton autotuner crashes on unknown autotune keys (e.g. STAGE, BT not in kernel args)
# Also fixes IndexError at runtime when _args is a subset of arg_names
python - <<'EOF'
import pathlib, site
p = pathlib.Path(site.getsitepackages()[0]) / 'triton/runtime/autotuner.py'
if p.exists():
    txt = p.read_text()
    patched = False
    old1 = 'self.key_idx = [arg_names.index(k) for k in key]'
    new1 = 'self.key_idx = [arg_names.index(k) for k in key if k in arg_names]'
    if old1 in txt:
        txt = txt.replace(old1, new1)
        patched = True
    old2 = 'key = [_args[i] for i in self.key_idx]'
    new2 = 'key = [all_args[self.arg_names[i]] for i in self.key_idx if self.arg_names[i] in all_args]'
    if old2 in txt:
        txt = txt.replace(old2, new2)
        patched = True
    if patched:
        p.write_text(txt)
        print('Patched: triton/runtime/autotuner.py')
EOF

