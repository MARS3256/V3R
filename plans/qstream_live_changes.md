# Solv3R live-streaming setup — what changed and what to focus on

Date: 2026-11-25. Scope: running Solv3R on `samples/drone1.mp4` / `drone2.mp4`, then on a
**live RTMP stream** that loops the video and is watchable in a browser.

## Starting point (original)

- `solv3r` is the upstream repo (`solv3r/`). Its README quick start is offline only: it takes a
  folder of images (or a video, via `list_images`) and solves all views together.
- The machine has a **GTX 1050 Ti, 4 GB VRAM**, Python 3.12 (no torch), no ffmpeg, no conda.

## What was added / changed

### New files
- `tests/qs.py` — offline quick start (README flow) with a commented CONFIG block at the top
  (input, stride, resolution, patch, calls, encode chunk, device, checkpoint, keep, sky, view_keep,
  html points, out dir) and per-frame timing logs. Writes `solve.npz` / `solve.ply` / `solve.html`.
- `tests/qstream.py` — **streaming** quick start (`solv3r.streaming.StreamingSolver`). Same config
  block plus the streaming window (`ANCHORS/RECENT/MEMORY/MEMORY_STRIDE/CALLS/ANCHOR_CALLS/
  REFRESH_TAU/REFRESH_CHECK`) and live options (`LIVE_MAX_SECONDS`, `LIVE_EXPORT_MAX`, `PREFETCH`).
  Accepts a file, a frames folder, **or a live URL** (`rtmp://`, `http(s)://`).
- `tests/serve_rtmp.py` — local **MediaMTX** RTMP server + a looping ffmpeg push to
  `rtmp://127.0.0.1:1935/live/drone`. Browser view: `http://127.0.0.1:8889/live/drone` (WebRTC)
  or `:8888` (HLS). The push re-encodes with `-bf 0` because WebRTC rejects H.264 with B-frames.

### Upstream edits
- `solv3r/solv3r/images.py` — `video_frames` used `ffmpeg -vsync 0`, removed in ffmpeg 7
  (imageio-ffmpeg 0.8.1 bundles 9.0.2). Changed to `-fps_mode passthrough` so passing an `.mp4`
  directly works.

### Environment fixes (not code)
- The venv's base Python 3.11 was a broken Microsoft Store stub. Installed real Python 3.11.9 to
  `C:\Python311` and repointed `venv\pyvenv.cfg` (`home`/`executable`). torch 2.7.1+cu118, CUDA on.
- Installed into the venv: `huggingface_hub`, `safetensors`, `imageio-ffmpeg`, `opencv-python`,
  `onnxruntime`.
- Copied the HF-cached weights to `checkpoints\solv3r.safetensors` and pointed `CHECKPOINT` at it,
  which removes the "unauthenticated requests to the HF Hub" warning (no Hub call at load).

## Key behavioural changes vs. the naive port

1. **Live mode** (`qstream.py`): a background `LiveReader` pulls frames from the stream into a small
   queue and **drops stale frames**, so the main loop stays live and processes the latest frames as
   fast as the GPU allows. `LIVE_MAX_SECONDS` bounds the run (default 60 s).
2. **Aspect-correct live frames**: live frames were initially forced to square 504×504 (1296 patches);
   now the stream is probed and scaled to the model size preserving aspect (280×504 = 720 patches).
   This roughly halved the anchor solve (61 s → 14.5 s).
3. **4 GB-safe config**: the paper's L8 (`MEMORY=64`, 504 px) overflows 4 GB VRAM and the driver
   **spills to shared system memory** (observed 3.8/4 GB VRAM + 11.6 GB shared), which is
   catastrophically slow. The defaults are now `LONG_SIDE=280`, `MEMORY=8`, `RECENT=4`,
   `CALLS=4`, `ANCHOR_CALLS=8`, plus `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`.
   Result: ~0.8 s/frame, stable (was 10–60 s/frame and climbing).
4. **Prefetch** (`PREFETCH=True`): next frame loads on a thread to overlap CPU loading with GPU work.
5. **Per-frame timing** logged for every frame (push ms), plus a final ms/frame summary.

## Config diff: ~9000 ms/frame → ~800 ms/frame

Only the config changed (same `StreamingSolver`). `PATCH` and `REFRESH_*` are unchanged.

| Config | Old (accurate, ~9 s/frame) | New (fast, ~0.8 s/frame) | Effect |
|---|---|---|---|
| `LONG_SIDE` | 504 | **280** | resolution ↓ ~2.5× fewer patches — **biggest accuracy loss** |
| `MEMORY` | 64 | **8** | held context ↓ — **big accuracy loss** (quadratic VRAM cost) |
| `RECENT` | 6 | 4 | small |
| `MEMORY_STRIDE` | 16 | 8 | small |
| `CALLS` | 8 | **4** | solver iterations ↓ — accuracy loss |
| `ANCHOR_CALLS` | 16 | 8 | anchor accuracy loss |
| `ANCHORS` | 8 | 8 | same |
| `PYTORCH_CUDA_ALLOC_CONF` | unset | `max_split_size_mb:128` | perf only (less fragmentation), no accuracy effect |

Why the old one was slow: the global-attention matrix is `(window × patches)²`. Old
`window≈79 × 720 patches` → ~100 GB, so it overflowed 4 GB and spilled to shared memory. New
`window≈21 × 220` → ~0.7 GB, fits.

### Accuracy vs. speed is a 2-D trade-off (resolution ↔ window)

The two accuracy levers are **resolution** (`LONG_SIDE`) and **context** (`MEMORY`/`CALLS`); both cost
VRAM quadratically. Balanced presets that still fit ~4 GB:

- **A — high-res, low-context:** `LONG_SIDE=504, MEMORY=4, RECENT=2, ANCHORS=4, CALLS=8, ANCHOR_CALLS=16`
  (sharp per-frame depth, weak global consistency).
- **B — balanced:** `LONG_SIDE=364, MEMORY=8, RECENT=4, ANCHORS=8, CALLS=6, ANCHOR_CALLS=12`
  (~2.6 GB attention; middle ground).
- **C — current fast:** `LONG_SIDE=280, MEMORY=8, RECENT=4, ANCHORS=8, CALLS=4, ANCHOR_CALLS=8`
  (~0.7 GB; fastest, least accurate).

To truly match the paper's L8 accuracy you need a larger GPU; on 4 GB these are the only options.

## Measured (GTX 1050 Ti, 4 GB)

- Offline 8 views, 16 calls: ~13 s solve (~0.83 s/call), 17 s total.
- Live, 4 GB-safe config: 32 frames in 26.6 s → **~0.83 s/frame**; periodic 1–3 s spikes are
  refresh calls (full-window re-solve), expected.

## What to focus on next

1. **Accuracy vs. speed is the main knob.** The 4 GB-safe config is an L8-equivalent, not the paper's
   L8. To recover accuracy on this GPU, raise `MEMORY`/`CALLS`/`LONG_SIDE` gradually and watch VRAM
   (keep it under ~3.5 GB to avoid shared-memory spill). Use `nvidia-smi` / Task Manager to confirm
   VRAM stays flat.
2. **Refresh tuning.** The 1–3 s spikes are refreshes. Increase `REFRESH_CHECK` / `REFRESH_TAU` for
   smoother streaming, decrease for tighter global consistency.
3. **Real-time vs. faster-than-real-time.** The solver is ~0.8 s/frame here vs. 25 fps input, so live
   mode drops frames. For a true 1:1 live demo you need a bigger GPU or the S4 preset
   (`ANCHORS=4, RECENT=4, MEMORY=16, CALLS=4`) at lower resolution.
4. **Browser latency.** WebRTC (`:8889`) is low-latency but needs `-bf 0` (already handled); HLS
   (`:8888`) is more compatible but ~10 s behind.
5. **Robustness.** `LiveReader` currently blocks on `queue.get()`; add a no-frame timeout so a dead
   stream exits with a clear error instead of hanging. `serve_rtmp.py` should pre-transcode the
   no-B-frame file once and `-c copy` loop it to avoid continuous CPU re-encode.
6. **Outputs.** `stream.npz` (c2w, c2w_exit, K, depth, depth_conf) is the real product; the `.ply` /
   `.html` are for inspection. For long streams, keep `LIVE_EXPORT_MAX` bounded.