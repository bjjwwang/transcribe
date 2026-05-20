# Deploying on a new GPU host

This is the bring-up checklist for putting `transcribe` on a fresh GPU box
behind a Cloudflare Tunnel. It assumes you've read the main `README.md`
and the operational gotchas section.

The whole thing takes ~30 min if HF model downloads cooperate.

---

## 0. Decide on a hostname

Pick the public hostname you'll serve it from, e.g. `transcribe.example.com`.
Cloudflare must already manage `example.com` (the zone has to be on your
Cloudflare account).

Decisions to make up front:

- **One tunnel or two?** If the new box already shares an existing tunnel
  with other services (i.e. you have a Cloudflare account with a tunnel
  configured locally for unrelated things), you can just add an `ingress`
  rule. Otherwise, create a fresh tunnel on the new box. Both flows are
  documented below — pick whichever applies before running through the
  rest of this doc.

- **Service password.** Pick something now, you'll put it in `.env`.

---

## 1. Host prereqs

### 1.1 Driver + GPU access

Confirm the box has a working NVIDIA driver and you can see the cards:

```bash
nvidia-smi
```

If this fails with "No devices were found" but the driver is loaded
(`lsmod | grep nvidia` shows entries, `/proc/driver/nvidia/gpus/` lists
the cards), it's almost always a **group permission** issue. Check:

```bash
id                       # is "video" in your groups?
ls -l /dev/nvidia0       # mode should be at least 0660 with you in the group
```

Fix (one of):

- `sudo usermod -aG video,render $USER`, then **log out and back in** so
  the new group memberships take effect.
- Or run the service as a user that's already in `video`.
- Avoid `chmod 0666 /dev/nvidia*` — it works until reboot, then comes
  back as `root:root 0660` and you've lost an hour figuring out why.

CUDA runtime version doesn't have to match what PyTorch was built against;
PyTorch ships its own CUDA libs. The driver just has to be ≥ that.

### 1.2 conda / mamba

If the box already has miniconda/mambaforge in your `$HOME` or a shared
path, fine — just verify it actually runs:

```bash
which conda
conda --version
```

If conda is broken because the install was moved (we hit `/srv -> /mnt`),
read "conda envs aren't relocatable" in the main `README.md`. The
TL;DR is: don't move it, or reinstall in place.

### 1.3 ffmpeg

WhisperX shells out to ffmpeg to read mp3/m4a. The conda env we build
below pulls a pinned ffmpeg in, so the system one doesn't matter — but
make sure no broken `ffmpeg` shadows it on `$PATH`.

### 1.4 Cloudflare account

You need a Cloudflare account with the target DNS zone added, and either:

- An existing tunnel + credentials file on the host (look for
  `~/.cloudflared/<uuid>.json`), or
- The ability to run `cloudflared tunnel login` interactively at least
  once to authorize a new tunnel.

---

## 2. Clone and install

```bash
# Pick a path; /mnt or /opt — anywhere your user can write that has a few
# GB free for HF model caches.
git clone https://github.com/bjjwwang/transcribe.git
cd transcribe

# Create the env — version pins matter, see "torch ABI" in README.
conda create -n whisperx -c conda-forge python=3.11 ffmpeg pip
ENV_BIN="$(conda info --base)/envs/whisperx/bin"

# torch / torchaudio / torchvision MUST all come from the same cu128 wheel
# index. If you skip torchvision and whisperx picks it up from PyPI later,
# you'll get a CPU build with mismatched ABI and torchvision will fail
# to import. Don't.
"$ENV_BIN/pip" install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0

# The rest. typing_extensions is normally satisfied by user-site; we
# install it explicitly because start.sh sets PYTHONNOUSERSITE=1.
"$ENV_BIN/pip" install \
  whisperx==3.8.5 "pyannote.audio==4.0.4" \
  fastapi uvicorn itsdangerous python-multipart typing_extensions
```

Smoke test CUDA from the env (run from outside `conda activate`, calling
the env's python directly — `conda activate` is fragile, see README):

```bash
PYTHONNOUSERSITE=1 "$ENV_BIN/python" -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('device count:', torch.cuda.device_count())
"
```

If `cuda available: False`, see step 1.1 — it's almost always /dev permissions.

---

## 3. Project config

### 3.1 `.env`

```bash
cp .env.example .env
# edit:
#   APP_PASSWORD=<the password you picked in step 0>
#   HF_TOKEN=hf_xxxxx     # required for diarization
#   COOKIE_SECURE=1       # since we're behind HTTPS via Cloudflare
chmod 600 .env
```

### 3.2 Accept the pyannote model terms

Once per Hugging Face account: go to
<https://huggingface.co/pyannote/speaker-diarization-community-1> and click
**Agree** on the gated-model dialog. The `HF_TOKEN` in `.env` must come
from the same account.

If you skip this, the first transcription job that reaches the diarize
stage fails with a 401 from `huggingface.co`. Easy to miss because the
upload and transcription stages succeed — error appears at the very end.

### 3.3 `start.sh`

Edit two paths in `start.sh`:

```bash
ENV_BIN=/path/to/your/env/bin    # whatever $ENV_BIN was above
```

`HF_HOME` / `TORCH_HOME` defaults under the project's `.cache/` are fine
unless that filesystem has a tight quota — in which case point them
somewhere with room (3+ GB).

### 3.4 First run, manually

```bash
./start.sh
```

It should print a couple of `transformers` deprecation warnings and then:

```
INFO:     Uvicorn running on http://127.0.0.1:8765
```

Ctrl-C to stop. If it falls over before that, the most likely causes are
in the operational notes section of `README.md` — go read those before
trying again.

---

## 4. Cloudflare Tunnel

### Option A — Add to an existing tunnel on this host

Edit your existing `~/.cloudflared/<config>.yml` and add an ingress rule
**before** the catch-all 404 row:

```yaml
ingress:
  # ...existing entries...
  - hostname: transcribe.example.com
    service: http://localhost:8765
  - service: http_status:404
```

Then add a CNAME for the new hostname:

```bash
cloudflared tunnel route dns <tunnel-name-or-uuid> transcribe.example.com
```

(You can also do this in the Cloudflare dashboard — DNS → Add record →
type CNAME, target `<tunnel-uuid>.cfargotunnel.com`, proxy ON.)

Then either restart the tunnel:

```bash
pkill -x cloudflared       # the supervisor (cron / systemd) will respawn it
```

…or, on newer cloudflared, send SIGHUP to reload without dropping
connections:

```bash
pkill -HUP -x cloudflared
```

### Option B — Create a new tunnel on this host

```bash
# Interactive browser flow once, leaves cert at ~/.cloudflared/cert.pem
cloudflared tunnel login

cloudflared tunnel create transcribe-prod
# prints a tunnel UUID and writes ~/.cloudflared/<uuid>.json

cloudflared tunnel route dns transcribe-prod transcribe.example.com
```

Write the config:

```yaml
# ~/.cloudflared/transcribe.yml
tunnel: <uuid printed above>
credentials-file: /home/<you>/.cloudflared/<uuid>.json

ingress:
  - hostname: transcribe.example.com
    service: http://localhost:8765
  - service: http_status:404
```

Run it:

```bash
cloudflared tunnel --config ~/.cloudflared/transcribe.yml run
```

For production, put that behind systemd or cron — see step 5.

### Note: 100 MB body limit on Free/Pro plans

If your Cloudflare account is on Free or Pro, single requests over
~100 MB still get a synthetic 413 from the edge. The frontend already
handles this via chunked upload (`/api/uploads/*`), so this won't break
real users — but **never raise the chunk threshold or chunk size in
`static/index.html` above 90 MB**. The current values (`CHUNK_SIZE = 80
MB`, `CHUNK_THRESHOLD = 90 MB`) are deliberate. Read the "Cloudflare's
100 MB body limit" section of `README.md`.

---

## 5. Supervisor

You need *something* to start `start.sh` on boot and restart it if it
crashes. Three patterns, pick one:

### 5.1 systemd user unit (cleanest)

```bash
mkdir -p ~/.config/systemd/user
cp deploy/transcribe.service ~/.config/systemd/user/
# Edit WorkingDirectory= and ExecStart= to match your install path

loginctl enable-linger $USER     # service runs without an active login
systemctl --user daemon-reload
systemctl --user enable --now transcribe
systemctl --user status transcribe
```

If you want auto-restart even on clean exit (e.g. you're keeping
`_idle_watcher`'s old `os._exit(0)` behaviour), edit the unit to set
`Restart=always` instead of `Restart=on-failure`.

### 5.2 cron-driven healthcheck (what we use)

A single `healthcheck.sh` that probes everything every 5 minutes, run from
crontab:

```cron
*/5 * * * * /path/to/healthcheck.sh
@reboot sleep 30 && /path/to/healthcheck.sh
```

Minimal `healthcheck.sh` block for transcribe (the full template lives in
the operational notes section of `README.md`):

```bash
if ! curl -s -o /dev/null http://localhost:8765/api/me --max-time 10; then
  pkill -f "uvicorn app:app" 2>/dev/null
  sleep 1
  cd /path/to/transcribe || exit 1
  setsid nohup ./start.sh >> logs/uvicorn.log 2>&1 < /dev/null &
  NEW_PID=$!
  for i in $(seq 1 15); do
    sleep 2
    curl -s -o /dev/null --max-time 2 http://localhost:8765/api/me && break
    kill -0 "$NEW_PID" 2>/dev/null || { echo "launcher died"; break; }
  done
fi
```

Don't change `>>` to `>` — that truncates the log on every restart and you
lose the previous failure's diagnostics.

Do the same for cloudflared:

```bash
if ! pgrep -x cloudflared > /dev/null; then
  nohup cloudflared tunnel --config ~/.cloudflared/transcribe.yml run \
    > /tmp/cloudflared.log 2>&1 &
fi
```

### 5.3 Just `nohup` (for a quick demo, not production)

```bash
nohup ./start.sh >> logs/uvicorn.log 2>&1 &
```

No restart, no boot-time start. Fine for "does this work at all" experiments.

---

## 6. Acceptance checklist

Walk through these in order from a different machine (not the GPU box):

```bash
# Tunnel up?
dig transcribe.example.com               # should resolve to CF IP

# Service reachable end to end?
curl -i https://transcribe.example.com/api/me
# expect: HTTP/2 200, body: {"authed":false}

# Auth works?
curl -i -c /tmp/c -X POST -F "password=<your APP_PASSWORD>" \
  https://transcribe.example.com/api/login
# expect: 200 with Set-Cookie ts_session=...

# Open the UI in a browser, log in, upload a small mp3, watch it transcribe.
# First-ever upload pulls down ~3 GB of models — be patient.

# Big file works (chunked path)?
# Upload a >100 MB mp3 from the browser. You should see the upload-progress
# percentage tick up smoothly; if you get "Upload failed: HTTP 413" with
# nothing useful behind it, double-check no nginx/caddy upstream is also
# applying a body limit smaller than 500 MB.

# GPU is actually being used?
curl -b /tmp/c https://transcribe.example.com/api/gpu | jq
# look for "cuda_available": true and process_view.*.allocated_mb > 0
# while a job is mid-transcription.
```

If `cuda_available` is `false` on this fresh box, you're back in the
"GPU permissions" / "user-site shadow" / "CUDA_VISIBLE_DEVICES=" maze
— go read the operational notes in `README.md`. Those are the only
things that ever cause this in practice.

---

## 7. Tearing down a host

To stop serving from this host (e.g. you've migrated to a different box
and want to retire the old one):

1. Remove the `transcribe.*` ingress entry from the cloudflared config
   on the host.
2. `pkill -x cloudflared` (or `pkill -HUP -x cloudflared` for hot reload);
   the supervisor will respawn it with the new config.
3. Optional: delete the DNS CNAME via `cloudflared tunnel route dns
   --overwrite-dns ...` is *not* how to remove it — use the Cloudflare
   dashboard or `cf-cli` to delete the DNS record.
4. Kill the local service: `pkill -f "uvicorn app:app"`.
5. Remove the transcribe block from `healthcheck.sh` so cron doesn't
   keep trying to revive it.
6. If you have `~/.cache/huggingface` and friends only on this box, you
   can leave them — they're harmless if the env is gone. Or `rm -rf
   .cache/` to reclaim the disk (~5 GB for large-v3 + pyannote).

That's it.
