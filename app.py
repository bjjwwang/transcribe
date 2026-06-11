"""FastAPI backend for the transcription web app."""

import asyncio
import os
import secrets
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import (
    Cookie, Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from itsdangerous import BadSignature, URLSafeTimedSerializer

import transcribe as ts_mod

ROOT = Path(__file__).parent.resolve()
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "output"
STATIC_DIR = ROOT / "static"
# Partial uploads sit here until the client calls finalize. The dir is
# under UPLOAD_DIR but uses an underscore prefix so it can't collide with
# a job_id (those are 12-hex uuids, no underscores).
UPLOAD_PARTIAL_DIR = UPLOAD_DIR / "_partial"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_PARTIAL_DIR.mkdir(exist_ok=True)

ts_mod.load_env()

PASSWORD = os.environ.get("APP_PASSWORD")
if not PASSWORD:
    raise RuntimeError("APP_PASSWORD missing from .env")
SESSION_SECRET = os.environ.get("SESSION_SECRET") or secrets.token_urlsafe(48)
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "0") == "1"
SESSION_COOKIE = "ts_session"
SESSION_MAX_AGE = 7 * 24 * 3600

ALLOWED_EXTS = {".mp3", ".m4a"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB

# Idle GPU release: if no job has touched the GPU for this long, drop any
# cached CUDA allocator memory back to the OS. The web server stays up; the
# next request reloads models lazily inside transcribe.run_pipeline.
IDLE_GPU_RELEASE_SECONDS = int(os.environ.get("IDLE_GPU_RELEASE_SECONDS", "1800"))

signer = URLSafeTimedSerializer(SESSION_SECRET, salt="ts-auth")

JOBS: dict[str, dict] = {}
# In-flight chunked uploads, keyed by upload_id. Lost on restart, which is fine —
# the frontend retries with a fresh upload_id, and stale partial files get reaped
# by the idle-watcher loop below.
UPLOADS: dict[str, dict] = {}
UPLOAD_TTL_SECONDS = 6 * 3600
WORKER_LOCK = asyncio.Semaphore(1)
LOGIN_FAIL_COUNT: dict[str, int] = {}
LAST_GPU_USE: float = 0.0
GPU_EVER_USED: bool = False

app = FastAPI(title="transcribe")


@app.on_event("startup")
async def _start_idle_watcher():
    asyncio.create_task(_idle_watcher())


def _reap_stale_uploads():
    """Delete partial uploads (and their tracking dict entries) that have
    been sitting around longer than UPLOAD_TTL_SECONDS without being
    finalized. Called once a minute from the idle-watcher."""
    now = time.time()
    for uid in list(UPLOADS.keys()):
        u = UPLOADS[uid]
        if now - u["created_at"] < UPLOAD_TTL_SECONDS:
            continue
        try:
            os.unlink(u["path"])
        except FileNotFoundError:
            pass
        UPLOADS.pop(uid, None)


async def _idle_watcher():
    global GPU_EVER_USED
    while True:
        await asyncio.sleep(60)
        _reap_stale_uploads()
        if not GPU_EVER_USED:
            continue
        if any(x["status"] in ("queued", "running") for x in JOBS.values()):
            continue
        if time.time() - LAST_GPU_USE < IDLE_GPU_RELEASE_SECONDS:
            continue
        print(
            f"[idle-watcher] idle > {IDLE_GPU_RELEASE_SECONDS}s, releasing cached VRAM",
            flush=True,
        )
        try:
            ts_mod.release_gpu_memory()
        except Exception as e:
            print(f"[idle-watcher] release_gpu_memory failed: {e}", flush=True)
        # Reset so we only re-release after the GPU is actually used again.
        GPU_EVER_USED = False


def issue_token() -> str:
    return signer.dumps("ok")


def is_authed(token: Optional[str]) -> bool:
    if not token:
        return False
    try:
        signer.loads(token, max_age=SESSION_MAX_AGE)
        return True
    except BadSignature:
        return False


def require_auth(ts_session: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE)):
    if not is_authed(ts_session):
        raise HTTPException(status_code=401, detail="auth required")


@app.post("/api/login")
async def login(password: str = Form(...)):
    if not secrets.compare_digest(password, PASSWORD):
        await asyncio.sleep(1.0)
        raise HTTPException(status_code=401, detail="wrong password")
    resp = JSONResponse({"ok": True})
    resp.set_cookie(
        SESSION_COOKIE, issue_token(),
        max_age=SESSION_MAX_AGE, httponly=True, samesite="strict", secure=COOKIE_SECURE,
        path="/",
    )
    return resp


@app.post("/api/logout")
async def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp


@app.get("/api/me")
async def me(ts_session: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE)):
    return {"authed": is_authed(ts_session)}


def _public_job(j: dict) -> dict:
    return {
        "id": j["id"],
        "filename": j["filename"],
        "status": j["status"],
        "stage": j["stage"],
        "progress": j["progress"],
        "started_at": j["started_at"],
        "finished_at": j["finished_at"],
        "queued_at": j["queued_at"],
        "error": j["error"],
        "queue_position": _queue_position(j),
    }


def _queue_position(j: dict) -> int:
    if j["status"] != "queued":
        return 0
    queued = [x for x in JOBS.values() if x["status"] == "queued"]
    queued.sort(key=lambda x: x["queued_at"])
    for i, x in enumerate(queued, start=1):
        if x["id"] == j["id"]:
            return i
    return 0


@app.post("/api/jobs", dependencies=[Depends(require_auth)])
async def create_job(
    file: UploadFile = File(...),
    speakers: Optional[int] = Form(default=None),
    model: str = Form(default="large-v3"),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"unsupported format: only {sorted(ALLOWED_EXTS)} allowed")

    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir()
    audio_path = job_dir / f"input{ext}"

    bytes_written = 0
    with audio_path.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            bytes_written += len(chunk)
            if bytes_written > MAX_UPLOAD_BYTES:
                out.close()
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(413, "file too large (max 500MB)")
            out.write(chunk)

    JOBS[job_id] = {
        "id": job_id,
        "filename": file.filename or f"audio{ext}",
        "status": "queued",
        "stage": "queued",
        "progress": 0.0,
        "queued_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "audio_path": str(audio_path),
        "output_path": str(OUTPUT_DIR / f"{job_id}.txt"),
        "speakers": speakers,
        "model": model,
        "size_bytes": bytes_written,
    }
    asyncio.create_task(run_job(job_id))
    return {"job_id": job_id}


# --- chunked upload (for files > Cloudflare's 100 MB body limit) ---------
# Three-step flow the frontend drives:
#   1) POST /api/uploads/init         -> upload_id (ext checked here)
#   2) PATCH /api/uploads/{upload_id} -> raw chunk body, X-Offset header
#   3) POST /api/uploads/{upload_id}/finalize -> creates a real job, returns job_id
# Each PATCH ships <100 MB so it slips past Cloudflare edge.

@app.post("/api/uploads/init", dependencies=[Depends(require_auth)])
async def upload_init(filename: str = Form(...)):
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"unsupported format: only {sorted(ALLOWED_EXTS)} allowed")
    upload_id = uuid.uuid4().hex[:16]
    path = UPLOAD_PARTIAL_DIR / f"{upload_id}{ext}"
    path.touch()
    UPLOADS[upload_id] = {
        "path": str(path),
        "ext": ext,
        "filename": filename,
        "size": 0,
        "created_at": time.time(),
    }
    return {"upload_id": upload_id}


@app.patch("/api/uploads/{upload_id}", dependencies=[Depends(require_auth)])
async def upload_chunk(
    upload_id: str,
    request: Request,
    x_offset: int = Header(..., alias="X-Offset"),
):
    u = UPLOADS.get(upload_id)
    if not u:
        raise HTTPException(404, "no such upload (it may have expired — retry)")
    if x_offset < 0 or x_offset > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "bad offset")
    end = x_offset
    # Stream the chunk straight to disk at the requested offset. Starlette's
    # request.stream() yields small pieces (~16-64KB) so peak RAM stays tiny
    # even for an 80 MB chunk.
    with open(u["path"], "r+b") as out:
        out.seek(x_offset)
        async for piece in request.stream():
            if not piece:
                continue
            out.write(piece)
            end += len(piece)
            if end > MAX_UPLOAD_BYTES:
                out.close()
                try:
                    os.unlink(u["path"])
                except FileNotFoundError:
                    pass
                UPLOADS.pop(upload_id, None)
                raise HTTPException(413, "file too large (max 500MB)")
    if end > u["size"]:
        u["size"] = end
    return {"bytes": u["size"]}


@app.post("/api/uploads/{upload_id}/finalize", dependencies=[Depends(require_auth)])
async def upload_finalize(
    upload_id: str,
    speakers: Optional[int] = Form(default=None),
    model: str = Form(default="large-v3"),
):
    u = UPLOADS.pop(upload_id, None)
    if not u:
        raise HTTPException(404, "no such upload (it may have expired — retry)")
    if u["size"] == 0:
        try:
            os.unlink(u["path"])
        except FileNotFoundError:
            pass
        raise HTTPException(400, "no chunks received")

    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir()
    audio_path = job_dir / f"input{u['ext']}"
    shutil.move(u["path"], str(audio_path))

    JOBS[job_id] = {
        "id": job_id,
        "filename": u["filename"],
        "status": "queued",
        "stage": "queued",
        "progress": 0.0,
        "queued_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "audio_path": str(audio_path),
        "output_path": str(OUTPUT_DIR / f"{job_id}.txt"),
        "speakers": speakers,
        "model": model,
        "size_bytes": u["size"],
    }
    asyncio.create_task(run_job(job_id))
    return {"job_id": job_id}


@app.get("/api/gpu", dependencies=[Depends(require_auth)])
async def gpu_status():
    """Two views of GPU state, served from the uvicorn process which (unlike
    Claude's sandbox shell) can see the devices:
      - process_view: torch.cuda.memory_* — how much this uvicorn process owns
      - device_view : nvidia-smi — how much each card has used in total
    """
    import subprocess
    import torch

    out: dict = {"cuda_available": torch.cuda.is_available()}
    if out["cuda_available"]:
        proc = []
        for i in range(torch.cuda.device_count()):
            proc.append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "allocated_mb": round(torch.cuda.memory_allocated(i) / 1024 / 1024, 1),
                "reserved_mb": round(torch.cuda.memory_reserved(i) / 1024 / 1024, 1),
            })
        out["process_view"] = proc
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            rows = []
            for line in r.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    rows.append({
                        "index": int(parts[0]),
                        "vram_used_mb": int(parts[1]),
                        "vram_total_mb": int(parts[2]),
                        "util_pct": int(parts[3]),
                    })
            out["device_view"] = rows
        else:
            out["device_view_error"] = r.stderr.strip() or "nvidia-smi returned non-zero"
    except Exception as e:
        out["device_view_error"] = f"{type(e).__name__}: {e}"
    return out


@app.get("/api/jobs/{job_id}", dependencies=[Depends(require_auth)])
async def get_job(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "no such job")
    return _public_job(j)


@app.get("/api/jobs/{job_id}/result", dependencies=[Depends(require_auth)])
async def get_result(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "no such job")
    if j["status"] != "done":
        raise HTTPException(409, "job not done")
    out_name = Path(j["filename"]).stem + ".txt"
    return FileResponse(j["output_path"], media_type="text/plain; charset=utf-8", filename=out_name)


@app.get("/api/jobs/{job_id}/preview", dependencies=[Depends(require_auth)])
async def get_preview(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "no such job")
    if j["status"] != "done":
        raise HTTPException(409, "job not done")
    text = Path(j["output_path"]).read_text()
    return {"text": text}


async def run_job(job_id: str):
    global LAST_GPU_USE, GPU_EVER_USED
    j = JOBS[job_id]
    async with WORKER_LOCK:
        j["status"] = "running"
        j["stage"] = "starting"
        j["started_at"] = time.time()
        GPU_EVER_USED = True
        LAST_GPU_USE = j["started_at"]
        loop = asyncio.get_running_loop()

        def progress(stage: str, pct: float):
            j["stage"] = stage
            j["progress"] = float(pct)

        try:
            await loop.run_in_executor(None, lambda: ts_mod.run_pipeline_gpu(
                audio_path=j["audio_path"],
                output_path=j["output_path"],
                num_speakers=j["speakers"],
                progress_cb=progress,
            ))
            j["status"] = "done"
            j["stage"] = "done"
            j["progress"] = 1.0
        except Exception as e:
            j["status"] = "error"
            j["error"] = f"{type(e).__name__}: {e}"
        finally:
            j["finished_at"] = time.time()
            LAST_GPU_USE = j["finished_at"]
            try:
                shutil.rmtree(Path(j["audio_path"]).parent, ignore_errors=True)
            except Exception:
                pass


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
