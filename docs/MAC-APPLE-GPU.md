# Running on Apple Silicon (Mac, GPU-accelerated)

This branch adapts `transcribe` to run **fully on the Apple GPU** of an
Apple-Silicon Mac (M-series), instead of the upstream Linux/CUDA path.

## Why a separate path

The upstream pipeline uses **faster-whisper / CTranslate2** for ASR, which has
**no Metal (Apple GPU) backend** — on a Mac it can only run on CPU, which is
very slow, and it also wants a 3 GB CUDA-style model download. So on Mac we swap
the engines while keeping the same FastAPI app, auth, upload flow and output
format:

| Stage | Upstream (Linux/CUDA) | This branch (Mac) |
|-------|-----------------------|-------------------|
| ASR | faster-whisper `large-v3` (CUDA) | **MLX Whisper `large-v3`** (Apple GPU / Metal) |
| Word timestamps | whisperX wav2vec2 align | MLX `word_timestamps=True` |
| Diarization | pyannote (CUDA) | **pyannote on `mps`** (Apple GPU, with CPU fallback) |

On an M4 Max, a 32-minute clip goes from ~2–4 h (CPU) to **~3.5 min** end-to-end,
at `large-v3` quality with speaker labels.

The new code is additive: `transcribe.run_pipeline()` (the CUDA path) is
untouched; this branch adds `transcribe.run_pipeline_gpu()` and points the web
app at it.

## Install

```bash
# Python env (3.11) via uv; ffmpeg via Homebrew.
brew install ffmpeg
uv venv --python 3.11 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install \
  torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0        # default PyPI = Mac arm64 CPU wheels
VIRTUAL_ENV=$PWD/.venv uv pip install \
  whisperx==3.8.5 "pyannote.audio==4.0.4" mlx-whisper \
  fastapi uvicorn itsdangerous python-multipart typing_extensions
```

## Models

Place the MLX model under `models/` (gitignored):

```bash
huggingface-cli download mlx-community/whisper-large-v3-mlx \
  --local-dir models/whisper-large-v3-mlx
```

Override the location with `MLX_WHISPER_MODEL=/path/to/model` if needed.
Diarization still uses the gated `pyannote/speaker-diarization-community-1`
(accept its terms on Hugging Face; set `HF_TOKEN` in `.env`).

## Config / run

```bash
cp .env.example .env     # set APP_PASSWORD, HF_TOKEN, COOKIE_SECURE
./start-mac.sh           # uvicorn on 127.0.0.1:8765
```

`start-mac.sh` is the Mac launcher (uv venv, Homebrew ffmpeg,
`PYTORCH_ENABLE_MPS_FALLBACK=1` so pyannote can use MPS).

## Idle memory release

MLX pins the loaded model (~3 GB) in unified memory across requests for speed.
`release_gpu_memory()` drops it (`ModelHolder.model = None` + `mx.clear_cache()`
+ `torch.mps.empty_cache()`), and the app's existing `_idle_watcher` calls it
after `IDLE_GPU_RELEASE_SECONDS` of no activity (default 1800; we run 600). The
next request reloads the model lazily (~2–3 s cold start).

## Standalone CLI

`run_full.py` runs the same GPU pipeline outside the web app:

```bash
HF_TOKEN=hf_... ./.venv/bin/python run_full.py input.m4a output.txt
```
