#!/usr/bin/env bash
# Start the transcription web app on 127.0.0.1:8765 (single worker, GPU-bound).
set -euo pipefail
cd "$(dirname "$0")"

# whisperx env lives directly on /mnt/scratch. Don't `conda activate` —
# the base conda's internal paths are still pinned to the old /srv prefix
# and it just errors out. Calling the env's interpreters directly works fine.
ENV_BIN=/mnt/scratch/PAG/Wjw/miniconda3/envs/whisperx/bin
export PATH="$ENV_BIN:$PATH"

# Sandbox / cron parents sometimes export CUDA_VISIBLE_DEVICES="" (empty
# string) which hides every GPU from torch. We always want all 3 L4s here,
# so just force-clear both vars unconditionally. If you ever need to pin a
# specific card, set it after start.sh launches uvicorn, not before.
unset CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES

# Block ~/.local/lib/python*/site-packages from leaking into this env. A stray
# torch 2.5.1+cu121 lives there and otherwise shadows our env's torch 2.8.0+cu128,
# breaking torchaudio/torchvision with undefined-symbol errors.
export PYTHONNOUSERSITE=1

# Set COOKIE_SECURE=1 once you're behind HTTPS.
export COOKIE_SECURE="${COOKIE_SECURE:-1}"

# Redirect model/library caches to /mnt/scratch — /home has a per-user NFS quota
# that whisperx + pyannote downloads will quickly exceed (EDQUOT).
export HF_HOME="${HF_HOME:-/mnt/scratch/PAG/Wjw/transcribe/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-/mnt/scratch/PAG/Wjw/transcribe/.cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/scratch/PAG/Wjw/transcribe/.cache/xdg}"
export PYANNOTE_CACHE="${PYANNOTE_CACHE:-/mnt/scratch/PAG/Wjw/transcribe/.cache/pyannote}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$PYANNOTE_CACHE"

exec "$ENV_BIN/uvicorn" app:app \
  --host 127.0.0.1 --port 8765 \
  --workers 1 \
  --timeout-keep-alive 75 \
  --log-level info
