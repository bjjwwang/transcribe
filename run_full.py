"""全 GPU 流程:MLX 转写(Apple GPU)+ pyannote 分离(MPS)+ 合并出带说话人标签的稿。"""
import os, sys, time
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import mlx_whisper
import whisperx
from whisperx.diarize import DiarizationPipeline
import transcribe as ts  # 复用 merge/normalize/render

AUDIO = sys.argv[1]
OUT = sys.argv[2]
_HERE = os.path.dirname(os.path.abspath(__file__))
MLX_MODEL = os.environ.get("MLX_WHISPER_MODEL") or os.path.join(_HERE, "models", "whisper-large-v3-mlx")
HF_TOKEN = os.environ["HF_TOKEN"]

T = time.time()

print("[1/4] MLX 转写(Apple GPU,带词级时间戳)...", flush=True)
t0 = time.time()
r = mlx_whisper.transcribe(AUDIO, path_or_hf_repo=MLX_MODEL, language="en",
                           word_timestamps=True, verbose=False)
print(f"   转写完成 {time.time()-t0:.0f}s, {len(r['segments'])} 段", flush=True)

# 转成 whisperx 期望的结构
segments = []
for s in r["segments"]:
    words = [{"word": w["word"], "start": w.get("start"), "end": w.get("end")}
             for w in s.get("words", []) if w.get("start") is not None]
    segments.append({"start": s["start"], "end": s["end"], "text": s["text"], "words": words})
result = {"segments": segments}

print("[2/4] 载入音频做分离...", flush=True)
audio = whisperx.load_audio(AUDIO)

print("[3/4] pyannote 说话人分离(Apple GPU / MPS)...", flush=True)
t0 = time.time()
diarize = DiarizationPipeline(model_name="pyannote/speaker-diarization-community-1",
                              token=HF_TOKEN, device="mps")
diarize_segments = diarize(audio)
print(f"   分离完成 {time.time()-t0:.0f}s", flush=True)

print("[4/4] 合并说话人标签 + 渲染...", flush=True)
result = whisperx.assign_word_speakers(diarize_segments, result)
blocks = ts.merge_into_speaker_blocks(result["segments"])
blocks = ts.normalize_speaker_labels(blocks)
text = ts.render(blocks)
with open(OUT, "w") as f:
    f.write(text)

nspk = len(set(b["speaker"] for b in blocks))
print(f"\n✅ 全部完成,总耗时 {time.time()-T:.0f}s | {nspk} 个说话人 | {len(blocks)} 个发言块 -> {OUT}", flush=True)
