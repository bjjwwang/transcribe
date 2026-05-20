# transcribe

A small self-hosted web app that turns long Chinese/English audio (mp3, m4a)
into speaker-labelled transcripts. Upload through the browser, get back text
with `Speaker 1: ...` blocks.

The pipeline is [WhisperX](https://github.com/m-bain/whisperX) (faster-whisper
+ alignment) followed by [pyannote.audio](https://github.com/pyannote/pyannote-audio)
speaker diarization. The web layer is FastAPI + a single static HTML page,
fronted by a Cloudflare Tunnel so the box doesn't need a public IP.

This is **not a product**. It runs on one GPU box, takes one user at a time,
and is sized for our own use. The interesting bits are in the deployment
notes below — most of those were paid for in production outages.

## Architecture

```
                    +-----------------+
   browser  --->    | Cloudflare      |
 (any IP, any net)  | tunnel + TLS    |
                    +--------+--------+
                             |  HTTP/2 (100 MB body limit on Free/Pro!)
                             v
                    +-----------------+      single uvicorn worker
                    | cloudflared     |      (asyncio.Semaphore(1) gates the GPU)
                    | (localhost only)|
                    +--------+--------+
                             |
                             v
                    +-----------------+
                    | FastAPI         |
                    |   /api/login    |
                    |   /api/jobs     |  small uploads (<90 MB)
                    |   /api/uploads/ |  chunked uploads (≥90 MB, see "Cloudflare 100 MB")
                    |   /api/gpu      |
                    +--------+--------+
                             |
                             v
                    +-----------------+
                    | transcribe.py   |  WhisperX -> align -> pyannote diarize
                    | run_pipeline    |  releases VRAM in finally{}
                    +--------+--------+
                             |
                             v
                    NVIDIA L4 (one card; the box has three but we pin none)
```

State is in-process dicts (`JOBS`, `UPLOADS`) — restart loses queued work,
but finished `.txt` outputs are on disk. There's no DB.

## Layout

```
app.py             FastAPI app, auth, upload endpoints, job lifecycle, idle-watcher
transcribe.py     WhisperX + pyannote pipeline (run_pipeline) and CLI entry point
start.sh           launcher used by cron / systemd; sets cache dirs and env safety
static/index.html  single-page UI (vanilla JS, no build step)
deploy/            reference nginx.conf, Caddyfile, transcribe.service (templates)
.env.example       copy to .env and fill in
```

## Running it

### Prereqs

- NVIDIA GPU with CUDA 12.x driver (we use 3× L4, but one is plenty)
- A user account in the `video` group (see "GPU permissions" below)
- conda / mamba
- A Cloudflare Tunnel pointing `<your host>` -> `http://localhost:8765`
  (not strictly required; can also expose 8765 via nginx/caddy — see `deploy/`)

### Install

```bash
# 1. Create the env. The exact wheel sources matter — see "torch + torchaudio
#    + torchvision ABI" in the gotchas section before tweaking versions.
conda create -n whisperx -c conda-forge python=3.11 ffmpeg pip
ENV=$CONDA_PREFIX/envs/whisperx
"$ENV/bin/pip" install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0
"$ENV/bin/pip" install \
  whisperx==3.8.5 "pyannote.audio==4.0.4" \
  fastapi uvicorn itsdangerous python-multipart typing_extensions

# 2. Config
cp .env.example .env
# edit .env — set APP_PASSWORD, HF_TOKEN, optionally SESSION_SECRET

# 3. Accept the pyannote model terms on Hugging Face (once per HF account):
#    https://huggingface.co/pyannote/speaker-diarization-community-1

# 4. Edit start.sh — point ENV_BIN at your env's bin/ directory if it isn't
#    /mnt/scratch/PAG/Wjw/miniconda3/envs/whisperx/bin

# 5. Start
./start.sh
```

The first request triggers WhisperX to download `large-v3` (~3 GB) and the
pyannote checkpoint. They go to `$HF_HOME` / `$TORCH_HOME` (set by `start.sh`),
so they survive process restarts.

### Health-check / supervisor

We run `start.sh` from a cron-driven `healthcheck.sh` (see the `deploy/`
section in *Operational notes* below). systemd would be cleaner if you have
linger / a real service account. Pick one.

## Endpoints

| Method | Path                                  | Notes                                        |
|--------|---------------------------------------|----------------------------------------------|
| POST   | `/api/login`                          | form: `password` — sets `ts_session` cookie  |
| POST   | `/api/logout`                         |                                              |
| GET    | `/api/me`                             | `{authed: bool}`                             |
| POST   | `/api/jobs`                           | multipart, single-shot upload (≤ ~90 MB body)|
| POST   | `/api/uploads/init`                   | form: `filename` -> `{upload_id}`            |
| PATCH  | `/api/uploads/{upload_id}`            | raw body, header `X-Offset: <bytes>`         |
| POST   | `/api/uploads/{upload_id}/finalize`   | form: `model`, `speakers` -> `{job_id}`      |
| GET    | `/api/jobs/{job_id}`                  | status / stage / progress                    |
| GET    | `/api/jobs/{job_id}/result`           | full text                                    |
| GET    | `/api/jobs/{job_id}/preview`          | first ~few hundred lines                     |
| GET    | `/api/gpu`                            | torch.cuda + nvidia-smi snapshot             |

All `/api/*` (except `/api/login` and `/api/me`) require the session cookie.

The front-end picks between `/api/jobs` and the `/api/uploads/*` chunked
flow based on file size — see "Cloudflare's 100 MB body limit" below.

---

# Operational notes — the deployment gotchas

Every bullet here was a production outage at some point. If you're touching
the deploy, read this section first; you'll save yourself an evening.

## 1. Cloudflare's 100 MB request-body limit

**Symptom:** the browser shows `上传失败：` (or `Upload failed:`) with **no
status code, no detail** — just empty after the colon. The server logs see
no request at all for that upload.

**Cause:** Cloudflare's Free and Pro plans hard-cap a single HTTP request
body at 100 MB. Anything bigger gets a synthetic `413 Payload Too Large`
returned by the edge before a byte reaches your origin. Enterprise lifts
this to 500 MB and can go higher; Free/Pro can't be configured around.

Diagnosing this from the front-end was complicated by two more facts:

- **HTTP/2 has no reason-phrase**, so `Response.statusText` is the empty
  string when traffic goes through Cloudflare. Lazily writing
  `msg = res.statusText` gives you nothing to show the user.
- Cloudflare's 413 body is an HTML error page, not JSON, so `res.json()`
  throws and a naive `try { ... } catch {}` swallows it. You're left with
  an empty error string.

**Fix (client side):** in any error path, lead with the numeric status:

```js
let msg = `HTTP ${res.status}` + (res.statusText ? " " + res.statusText : "");
try { const j = await res.json(); if (j.detail) msg = j.detail; } catch {}
throw new Error(msg);
```

**Fix (real fix — transparent chunked upload):** when a picked file is
above ~90 MB, the client takes a different path:

1. `POST /api/uploads/init` with the filename — gets back `upload_id`.
2. For each 80 MB slice of the file, `PATCH /api/uploads/{upload_id}`
   with the raw bytes as body and an `X-Offset: <bytes>` header.
3. `POST /api/uploads/{upload_id}/finalize` with `model` / `speakers` —
   creates a real job, returns `job_id`. The pipeline runs as if the file
   had been uploaded in one shot.

The server writes each PATCH directly to disk at the given offset, so peak
RAM stays tiny even at 80 MB chunks. `MAX_UPLOAD_BYTES = 500 MB` is the
hard ceiling. A stale-upload reaper deletes partials older than 6 hours.

```
client                       server (uvicorn)
  |  POST /api/uploads/init     -> {upload_id}
  |  PATCH /api/uploads/<id>    open(path, "r+b") .seek(0)      .write(80MB)
  |     header X-Offset: 0
  |  PATCH /api/uploads/<id>    open(path, "r+b") .seek(80MB)   .write(rest)
  |     header X-Offset: 83886080
  |  POST /api/uploads/<id>/finalize  shutil.move(partial, jobs/<jid>/input.mp3)
  |                                   asyncio.create_task(run_job(jid))
```

The UI shows `上传中... NN%` so the user gets feedback for big files.
Below the threshold (~90 MB), the original `/api/jobs` single-shot path
is used — zero regression for small uploads.

## 2. PATH/env that breaks GPU access

A grab-bag of issues we hit when moving from one host to another or
restarting the box. Each of them silently fell back to CPU or refused to
boot.

### 2.1 `CUDA_VISIBLE_DEVICES=""` (empty string) hides every GPU

Some parent shells (web terminals, certain sandboxes) export
`CUDA_VISIBLE_DEVICES` set to the empty string, which CUDA treats as
"no devices visible". The fix is to **unconditionally clear it** in
`start.sh`:

```bash
unset CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES
```

Beware: the bash idiom `[ -z "${VAR:-x}" ]` will **not** detect "set to
empty string" — the `:-` operator treats empty the same as unset. Use
`[ "${VAR-x}" = "" ]` if you need to distinguish, or just `unset`
unconditionally.

### 2.2 `~/.local/lib/python3.11/site-packages/` shadowing the env

If anything in your home dir's user-site has a different `torch`
(common: a stale `torch==2.5.1+cu121` from an old project), Python's
import order puts it **before** your conda env's torch. Then the env's
torchaudio/torchvision — built against the env's torch ABI — fail at
import with errors like:

```
undefined symbol: _ZNK5torch8autograd4Node4nameB5cxx11Ev
```

Or, more subtly:

```
AttributeError: partially initialized module 'torchvision' has no
attribute 'extension' (most likely due to a circular import)
```

`start.sh` sets `PYTHONNOUSERSITE=1` to kill user-site for this process.
Bonus consequence: any packages your env was *implicitly* getting from
user-site (we hit `typing_extensions`) will be missing. Install them
explicitly into the env.

### 2.3 GPU permissions (`/dev/nvidia*` is `root:root 0660`)

On a fresh reboot of a multi-tenant box, `/dev/nvidia0,1,2` come up as
`crw-rw---- root:root`. Only members of `video` (and `render`) can open
them. If your user isn't in `video`, PyTorch reports:

```
CUDA initialization: CUDA unknown error
Setting the available devices to be zero.
```

Check with:

```bash
id                    # is "video" in your groups?
ls -l /dev/nvidia0    # mode and owner
python3 -c "import os; os.open('/dev/nvidia0', os.O_RDWR)"  # EACCES?
```

Fix is one of: ask sysadmin to `usermod -aG video <user>` (permanent),
or `chmod 0666 /dev/nvidia*` (temporary, lost on reboot), or run the
service as a user that's already in `video`.

If `nvidia-smi` itself prints `No devices were found` from your shell
but works for another user — that's the same problem, not "the GPUs
are gone". Hardware is still healthy; `/proc/driver/nvidia/gpus/` lists
the cards regardless of permissions.

## 3. conda envs aren't relocatable

If the conda install gets moved (we had a `/srv -> /mnt` filesystem
migration), `bin/conda` still has `#!/srv/.../python` baked into its
shebang and refuses to run. `source $CONDA_PREFIX/etc/profile.d/conda.sh`
also exports the old prefix internally.

Workarounds, easiest first:

- **Don't `conda activate`.** Call the env's binaries by absolute path:
  `/path/to/env/bin/uvicorn` and friends. `start.sh` does this.
- `python3 -m conda <subcommand>` works for management because the
  Python interpreter itself isn't shebang-broken; only the wrapper is.
- If you actually need `conda activate`, reinstall conda at the new path.

## 4. Idle GPU release — without taking the service down

WhisperX + pyannote together hold ~3 GB of VRAM after a job. We want to
return that to the driver between jobs so other tenants on the box can
use it, but **not** by killing the web service — that turns idle into
visible downtime.

The original implementation called `os._exit(0)` after 30 min idle and
relied on a cron `healthcheck.sh` to restart. In practice:

- The "release the CUDA context fully" benefit is small — PyTorch's
  context is ~300–500 MB per card and you only get that back by exiting.
- The cost is real outages: any blip in the restart path (env broken,
  shebang broken, user removed from `video` group) becomes a 502 for
  everyone until the next healthcheck tick.

Current behaviour (`_idle_watcher` in `app.py`): after `IDLE_GPU_RELEASE_SECONDS`
of no GPU use, we call `transcribe.release_gpu_memory()` (which is
`gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()`),
log it, and **keep the process up**. The caching allocator returns its
free pool to the driver; the residual ~300 MB per card stays until exit.

Per-job cleanup in `run_pipeline`'s `finally:` does the same call after
each transcription completes, so steady-state VRAM is "context overhead"
between jobs.

## 5. Cron-driven health check (`healthcheck.sh`)

We run a single `healthcheck.sh` every 5 minutes that probes a handful of
services and restarts any that are down. Live version of the relevant
block — copy this idiom for the next service:

```bash
if ! curl -s -o /dev/null http://localhost:8765/api/me --max-time 10; then
  pkill -f "uvicorn app:app" 2>/dev/null
  sleep 1
  cd /path/to/transcribe || exit 1
  # >> appends to keep the idle-watcher's history; setsid + </dev/null
  # detaches from cron's pseudo-terminal so the child survives the parent.
  setsid nohup ./start.sh >> logs/uvicorn.log 2>&1 < /dev/null &
  NEW_PID=$!
  # uvicorn + whisperx cold start is ~6 s — poll the port until it
  # responds; PID-still-alive is not the same as "port bound".
  for i in $(seq 1 15); do
    sleep 2
    if curl -s -o /dev/null --max-time 2 http://localhost:8765/api/me; then break; fi
    if ! kill -0 "$NEW_PID" 2>/dev/null; then
      echo "FAILED: launcher died before binding port"; break
    fi
  done
fi
```

Anti-patterns we hit:

- **`> logs/uvicorn.log`** (single `>`) — silently truncates the log
  every restart, losing the diagnostics from the previous failure.
  Always `>>` for restart-driven launches.
- **`sleep 2; kill -0 $PID`** — too short. WhisperX takes ~6 s to load
  modules and bind the port; PID being alive 2 s after launch means
  almost nothing. Poll the actual endpoint.
- **Multiple healthcheck instances colliding** — if you trigger it
  manually while a cron tick is running, both call `pkill ... uvicorn`
  and kill each other's just-launched processes. `flock` it if that
  matters; for us 5-minute ticks never collide naturally.

## 6. Don't lie to the user in error messages

The whole "Cloudflare 100 MB" debug session above lost a couple of hours
to `"上传失败：" + e.message` showing nothing because of two layered
defaults swallowing the actual cause. Lessons retained:

- **Always include the numeric HTTP status in the rendered error** — it's
  one extra word for the user and the difference between "investigate
  network" and "investigate CDN config".
- **If the response body isn't JSON, fall back to the status line** —
  don't let `res.json().detail` win when there's no JSON.
- **Log the request that just failed** — both client (DevTools network
  tab) and server (uvicorn's access log). If neither sees a record of
  the failing request, look upstream of both (CDN, tunnel, edge filter).

---

## License

Use at your own risk.
