# Context handoff — Solv3R live streaming (continue from here)

Date: 2026-11-25. Machine: Windows, **GTX 1050 Ti 4 GB**, venv at `D:\v3R\V3R-solv3r\venv`
(Python 3.11.9, torch 2.7.1+cu118, CUDA on).

## Goal

Run Solv3R on `samples/drone1.mp4` / `drone2.mp4` and on a **live RTMP stream** that loops the video
and is watchable in a browser, processing frames continuously.

## Current state (working)

- Offline quick start works: `tests/qs.py` → `outputs/qs/solve.{npz,ply,html}`.
- Live streaming works end-to-end:
  - `tests/serve_rtmp.py` starts MediaMTX + loops `drone1.mp4` to `rtmp://127.0.0.1:1935/live/drone`.
    Browser: `http://127.0.0.1:8889/live/drone` (WebRTC) or `:8888` (HLS).
  - `tests/qstream.py rtmp://127.0.0.1:1935/live/drone 1 outputs\qstream_live 60` reads the stream
    for up to 60 s and writes `stream.{npz,ply,html}`.
- 4 GB-safe config runs at **~0.8 s/frame** (was 10–60 s/frame). See `plans/qstream_live_changes.md`
  for the full change log and the config diff table.

## Key files

- `tests/qs.py` — offline; commented CONFIG block + per-frame timing.
- `tests/qstream.py` — streaming/live; CONFIG block incl. window + live knobs; `LiveReader` drops
  stale frames; aspect-correct live frames; prefetch; per-frame timing.
- `tests/serve_rtmp.py` — MediaMTX loop server (re-encodes `-bf 0` for WebRTC).
- `solv3r/solv3r/images.py` — patched `-vsync 0` → `-fps_mode passthrough` (ffmpeg 7).
- `checkpoints/solv3r.safetensors` — local weights (kills the HF Hub warning).
- `plans/qstream_live_changes.md` — what changed + focus areas.

## The core constraint

Global attention is `(window × patches)²` in VRAM. Old L8 (window≈79, 720 patches) ≈ 100 GB →
overflows 4 GB → driver spills to shared system memory → 9 s/frame. 4 GB-safe config (window≈21,
220 patches) ≈ 0.7 GB → fits → 0.8 s/frame. **Accuracy was traded for VRAM fit.**

Config knobs that matter: `LONG_SIDE` (resolution), `MEMORY` (context), `CALLS`/`ANCHOR_CALLS`
(iterations). `PATCH`, `REFRESH_*` unchanged. `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` is
perf-only.

## Current config (fast, less accurate) in `tests/qstream.py`

`LONG_SIDE=280, PATCH=14, ANCHORS=8, RECENT=4, MEMORY=8, MEMORY_STRIDE=8, CALLS=4,
ANCHOR_CALLS=8, REFRESH_TAU=0.02, REFRESH_CHECK=4, LIVE_MAX_SECONDS=60, LIVE_EXPORT_MAX=300,
PREFETCH=True, DEPTH=True, KEEP=1.0, SKY_MASK=False, DEVICE=cuda`.

## Next steps

1. **Improve accuracy** without spilling VRAM: try preset **B** `LONG_SIDE=364, MEMORY=8, RECENT=4,
   ANCHORS=8, CALLS=6, ANCHOR_CALLS=12` (~2.6 GB); watch VRAM stays < ~3.5 GB in Task Manager.
   Preset **A** `LONG_SIDE=504, MEMORY=4, RECENT=2, ANCHORS=4, CALLS=8, ANCHOR_CALLS=16` for sharper
   per-frame depth.
2. **Refresh tuning**: 1–3 s spikes are refresh full-window solves; raise `REFRESH_CHECK`/`REFRESH_TAU`
   for smoother output, lower for tighter consistency.
3. **Robustness**: add a no-frame timeout in `LiveReader` so a dead stream errors instead of hanging;
   `serve_rtmp.py` should pre-transcode a no-B-frame file once and `-c copy` loop it (avoid constant
   CPU re-encode).
4. **Real-time**: solver ~0.8 s/frame vs 25 fps input, so live drops frames; true 1:1 needs a bigger
   GPU or the S4 preset at lower resolution.

## How to run

```powershell
# terminal 1 (keep open): server + loop, browser http://127.0.0.1:8889/live/drone
.\venv\Scripts\python.exe tests\serve_rtmp.py

# terminal 2: offline
.\venv\Scripts\python.exe tests\qs.py samples\drone1.mp4 100 outputs\qs

# terminal 2: live (60 s budget)
.\venv\Scripts\python.exe tests\qstream.py rtmp://127.0.0.1:1935/live/drone 1 outputs\qstream_live 60
```

Note: background processes do not persist between agent tool calls here — start `serve_rtmp.py` in a
real terminal.