#!/usr/bin/env bash
# Mac (CPU) launcher for the transcribe web app -> http://127.0.0.1:8765
# The original start.sh targets a Linux/CUDA box with hardcoded conda paths;
# this version uses the local uv venv and Homebrew ffmpeg.
set -euo pipefail
cd "$(dirname "$0")"

ENV_BIN="$(dirname "$0")/.venv/bin"

# Make sure Homebrew ffmpeg is on PATH (whisperx shells out to it).
export PATH="/opt/homebrew/bin:$ENV_BIN:$PATH"

# No CUDA on a Mac; transcription runs on the Apple GPU via MLX, diarization on MPS.
unset CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES 2>/dev/null || true
# Let pyannote use the Apple GPU (MPS); unsupported ops fall back to CPU.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Keep HF / torch model caches inside the project (~3 GB on first run).
export HF_HOME="${HF_HOME:-$(pwd)/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$(pwd)/.cache/torch}"
export PYANNOTE_CACHE="${PYANNOTE_CACHE:-$(pwd)/.cache/pyannote}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$PYANNOTE_CACHE" logs

exec "$ENV_BIN/uvicorn" app:app \
  --host 127.0.0.1 --port 8765 \
  --workers 1 \
  --timeout-keep-alive 75 \
  --log-level info
