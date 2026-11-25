# V3R

V3R turns a drone video **or a live stream** into a metric-scale point cloud and depth maps
using [Solv3R](https://github.com/ubc-vision/solv3r). It ships two front ends over one core:

- **`v3r_cli.py`** — command-line reconstruction (the quickstart path).
- **Blender 5.2 add-on** (`V3R/`) — start/stop a background worker and watch the point cloud
  grow in the viewport while Blender stays interactive.

Inference runs in a **separate worker process** with its own Python environment. Blender never
imports `torch`; it only writes a JSON config, spawns the worker, and applies point-cloud
messages on a timer. Every failure is a verbose error — there is no silent fallback.

## What it produces

For each answered frame, under `samples/outputs/<run>/` (or the folder you choose):

```
depth/frame_000000.npz        float16 depth + confidence + c2w + K
depth_png/frame_000000.png    colourised depth
points/points_000000.npz      xyz float32 + rgb uint8
cumulative.npz                all points so far
solv3r_points.ply             cumulative cloud (MeshLab/CloudCompare)
latest.json                   last answered frame
meta.json                     config, model SHA256, device, torch version
report.md                     run summary, counts, mean/p95 latency
```

## Install

Requires Blender 5.2 and a Python 3.11–3.14 worker environment. The tested setup is the
system Python 3.14 with PyTorch ROCm; a CPU-only PyTorch works too.

```bash
python3 -m venv .venv-worker
.venv-worker/bin/pip install --upgrade pip

# PyTorch: ROCm build for AMD, or the CPU wheel on any machine.
.venv-worker/bin/pip install torch --index-url https://download.pytorch.org/whl/rocm7.2

.venv-worker/bin/pip install safetensors huggingface_hub pillow scipy opencv-python-headless numpy

git clone https://github.com/ubc-vision/solv3r.git third_party/solv3r
.venv-worker/bin/pip install -e third_party/solv3r --no-deps --ignore-requires-python

mkdir -p models
curl -L -o models/solv3r.safetensors \
  https://huggingface.co/jgkwak/solv3r/resolve/main/solv3r.safetensors
```

The model stays in `models/` — the worker never downloads into the Hugging Face cache.

On Windows the venv interpreter is `.venv-worker\Scripts\python.exe`. The add-on detects it
automatically when the **Worker Python** field is left empty; the CLI equivalent is
`.venv-worker\Scripts\python.exe v3r_cli.py ...`.

## CLI (quickstart)

```bash
# video file
.venv-worker/bin/python v3r_cli.py --source samples/drone1.mp4

# live stream (RTSP / RTMP / UDP / SRT / HTTP(S))
.venv-worker/bin/python v3r_cli.py --source rtmp://localhost/live/stream

# faster, lower-res, bounded run
.venv-worker/bin/python v3r_cli.py --source samples/drone1.mp4 \
  --preset S4 --image-long-side 252 --sample-fps 2 --max-frames 30
```

Useful flags: `--out`, `--device cpu|cuda`, `--preset S4|L4|L8`, `--image-long-side 252|504`,
`--sample-fps` (0 = every frame), `--max-frames`, `--point-stride`, `--point-budget`,
`--no-depth`, `--anchors/--recent/--memory/--calls` (advanced Solv3R knobs).

## Blender add-on

1. Install/copy the `V3R/` folder as a Blender add-on and enable it.
2. Open the 3D Viewport sidebar (`N`) → **V3R** tab.
3. Choose **Video File** or **Live Stream**, set the source.
4. Set **Output Folder** (default `samples/outputs`), **Device**, **Preset**, **Resolution**,
   **Sample FPS**, and optionally **Point Radius**.
5. Click **Start Reconstruction**. The point cloud (`V3R_PointCloud`) grows as frames are
   solved; **Stop** ends the worker and finalises `report.md`.
6. **Open Output** opens the run folder; **Clear Cloud** empties the viewport object.

The **Advanced** sub-panel exposes the full Solv3R streaming configuration: anchors, recent
frames, memory, memory stride, solver calls, anchor calls, refresh tau and refresh check.

## Stream notes (OBS, vdo.ninja)

The worker opens any URL that OpenCV/FFmpeg can decode: `rtsp://`, `rtmp://`, `udp://`,
`srt://`, `http://`, `https://`. Examples:

- OBS → **Settings → Stream → Service: Custom** → `rtmp://localhost/live/stream` (with an
  RTMP server such as MediaMTX), then use that URL in V3R.
- OBS → **Start Virtual Camera** or an NDI/RTSP bridge, then use the bridge URL.
- **vdo.ninja** is WebRTC, not a direct video URL. Capture the vdo.ninja browser source in
  OBS and output it as RTMP/RTSP/HTTP; give V3R that output URL. A raw `https://vdo.ninja/...`
  page URL is not a video stream and will be rejected with a verbose error.

For live input, V3R samples at `Sample FPS` (default 2) and drops frames it cannot keep up
with. If no frame arrives within `Stream Lost Timeout`, the run stops with `StreamLostError`.

## Performance and backend

- **Backend is a startup setting.** `--device cpu` (default) or `--device cuda`. If the
  requested backend is unavailable, the worker exits with a verbose error; it does not
  silently fall back.
- **This machine (AMD RX 5700 XT, gfx1010):** ROCm 7.2 has no gfx1010 binaries, so the ROCm
  wheel cannot execute kernels. The configured `cpu` backend is used. Measured at
  `S4 / 252 px`: ~0.3–0.7 s per answered frame, which is fast enough for progressive updates.
- **For live use:** `S4 / 252 px / 2 fps` is the responsive default. For offline quality,
  use `L8 / 504 px / 0 fps`.
- **Point cloud rendering:** native Blender Point Cloud object with per-point `radius` and
  `Col` colour. EEVEE and Cycles both render it without duplicating geometry per point.

## Packaging

Two archives are produced at the repo root:

- **`V3R_addon.zip`** — the Blender add-on folder only (`V3R/`). Install it with
  **Edit → Preferences → Add-ons → Install from Disk**. It contains no worker or model.
- **`V3R_bundle.zip`** — the add-on plus `worker/`, `v3r_cli.py`, docs, plans and the two
  sample clips. It excludes `.venv*`, the 448 MB model, and `samples/outputs/`.

The add-on resolves the worker interpreter, `worker/app.py`, the model and the output folder
relative to the repository root. If you install `V3R_addon.zip` outside a checkout, set those
four paths explicitly in the add-on's **Advanced** sub-panel (absolute paths).

## Development

```bash
.venv/bin/pip install -e .[dev]          # ruff, mypy, pytest
.venv/bin/ruff format core worker tests
.venv/bin/ruff check core worker tests
.venv/bin/mypy core worker tests
.venv/bin/pytest
```

The worker has its own environment (`.venv-worker`) because it owns PyTorch/Solv3R.
Blender-side code must not import `torch`, `onnxruntime`, or the worker package.

## License

GPL-3.0-or-later. Solv3R is Apache-2.0; its NVIDIA weights are research-only.