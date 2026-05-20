"""
Transcribe an audio file (mp3/m4a/wav/...) into a speaker-labeled document.

Output format:

    Transcript

    00:00:01 Speaker 1
    Great.

    00:00:03 Speaker 2
    I might need to as well.

CLI:
    python transcribe.py path/to/audio.m4a
    python transcribe.py path/to/audio.m4a --model large-v3 --speakers 2

Library:
    from transcribe import run_pipeline
    run_pipeline(audio_path, output_path, num_speakers=2, progress_cb=fn)
"""

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Callable, Optional

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline


def release_gpu_memory():
    """Drop refs and ask CUDA to return cached VRAM to the OS."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def fmt_ts(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def merge_into_speaker_blocks(segments):
    blocks = []
    for seg in segments:
        spk = seg.get("speaker", "Unknown")
        text = seg.get("text", "").strip()
        start = seg.get("start", 0.0)
        if not text:
            continue
        if blocks and blocks[-1]["speaker"] == spk:
            blocks[-1]["lines"].append(text)
        else:
            blocks.append({"speaker": spk, "start": start, "lines": [text]})
    return blocks


def normalize_speaker_labels(blocks):
    mapping = {}
    next_idx = 1
    for b in blocks:
        raw = b["speaker"]
        if raw not in mapping:
            mapping[raw] = f"Speaker {next_idx}"
            next_idx += 1
        b["speaker"] = mapping[raw]
    return blocks


def render(blocks) -> str:
    out = ["Transcript", ""]
    for b in blocks:
        out.append(f"{fmt_ts(b['start'])} {b['speaker']}")
        for line in b["lines"]:
            out.append(line)
        out.append("")
    return "\n".join(out).rstrip() + "\n"


ProgressCb = Optional[Callable[[str, float], None]]


def _emit(cb: ProgressCb, stage: str, pct: float):
    if cb is not None:
        try:
            cb(stage, pct)
        except Exception:
            pass


def run_pipeline(
    audio_path: str,
    output_path: str,
    *,
    model: str = "large-v3",
    language: str = "en",
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    device: Optional[str] = None,
    diarize_model: str = "pyannote/speaker-diarization-community-1",
    hf_token: Optional[str] = None,
    progress_cb: ProgressCb = None,
) -> str:
    """Run the full transcription+diarization pipeline. Returns the rendered text."""
    load_env()
    hf_token = hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not set")

    audio_path = str(Path(audio_path).resolve())
    if not Path(audio_path).exists():
        raise FileNotFoundError(audio_path)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    compute_type = "float16" if device == "cuda" else "int8"
    batch_size = 16 if device == "cuda" else 4

    asr_model = align_model = diarize = None
    audio = None
    try:
        _emit(progress_cb, "loading_audio", 0.02)
        audio = whisperx.load_audio(audio_path)

        _emit(progress_cb, "transcribing", 0.05)
        asr_model = whisperx.load_model(model, device, compute_type=compute_type, language=language)

        def asr_progress(p):
            _emit(progress_cb, "transcribing", 0.05 + (float(p) / 100.0) * 0.50)

        result = asr_model.transcribe(
            audio, batch_size=batch_size, language=language, progress_callback=asr_progress
        )
        del asr_model; asr_model = None
        if device == "cuda":
            torch.cuda.empty_cache()

        _emit(progress_cb, "aligning", 0.55)
        align_model, metadata = whisperx.load_align_model(language_code=language, device=device)

        def align_progress(p):
            _emit(progress_cb, "aligning", 0.55 + (float(p) / 100.0) * 0.20)

        result = whisperx.align(
            result["segments"], align_model, metadata, audio, device,
            return_char_alignments=False, progress_callback=align_progress,
        )
        del align_model; align_model = None
        if device == "cuda":
            torch.cuda.empty_cache()

        _emit(progress_cb, "diarizing", 0.78)
        diarize = DiarizationPipeline(model_name=diarize_model, token=hf_token, device=device)
        diarize_kwargs = {}
        if num_speakers is not None:
            diarize_kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            diarize_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarize_kwargs["max_speakers"] = max_speakers
        diarize_segments = diarize(audio, **diarize_kwargs)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        del diarize; diarize = None

        _emit(progress_cb, "writing", 0.97)
        blocks = merge_into_speaker_blocks(result["segments"])
        blocks = normalize_speaker_labels(blocks)
        text = render(blocks)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)

        _emit(progress_cb, "done", 1.0)
        return text
    finally:
        del asr_model, align_model, diarize, audio
        release_gpu_memory()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="Path to audio file (mp3/m4a/wav/...)")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--language", default="en")
    ap.add_argument("--speakers", type=int, default=None)
    ap.add_argument("--min-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=None)
    ap.add_argument("--output-dir", default=str(Path(__file__).parent / "output"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--diarize-model", default="pyannote/speaker-diarization-community-1")
    args = ap.parse_args()

    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        print(f"ERROR: file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output_dir) / (audio_path.stem + ".txt")

    def cb(stage, pct):
        print(f"[{int(pct*100):3d}%] {stage}")

    run_pipeline(
        str(audio_path), str(out_path),
        model=args.model, language=args.language,
        num_speakers=args.speakers, min_speakers=args.min_speakers, max_speakers=args.max_speakers,
        device=args.device, diarize_model=args.diarize_model,
        progress_cb=cb,
    )
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
