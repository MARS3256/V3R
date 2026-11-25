"""Solv3R streaming quick start on a drone video or a live stream, with per-frame timing logged.

Every frame is answered when it arrives (streaming mode), in the frame of the first frame. The input
is either a video file / frames folder (as in qs.py) or a live stream (rtmp://, http(s)://): in live
mode frames are read from the stream for up to LIVE_MAX_SECONDS and processed continuously.

Writes <OUT_DIR>/stream.npz (camera-to-world on arrival and when a frame left the recent window,
intrinsics, depth on arrival), <OUT_DIR>/stream.ply (point cloud) and <OUT_DIR>/stream.html.

usage: python tests/qstream.py [video|frames_folder|rtmp_url] [stride] [out_dir] [live_seconds]
"""
import os
import sys
import threading
import time
import queue
import subprocess

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")  # less fragmentation on a 4 GB GPU

# --------------------------------------------------------------------------------------------------
# CONFIGS — every knob of the Solv3R streaming quick start. Edit here, or override the first three
# on the CLI. The ANCHORS/RECENT/MEMORY/... fields are the StreamConfig of solv3r.streaming.
# --------------------------------------------------------------------------------------------------
INPUT = r"D:\v3R\V3R-solv3r\samples\drone1.mp4"  # a video file, a frames folder, OR a live stream URL
                        # (rtmp://127.0.0.1:1935/live/drone, http://...). A URL switches to live mode.
STRIDE = 1              # (file/folder only) use every STRIDE-th frame.
MAX_IMAGES = None       # (file/folder only) cap the number of frames (None = no cap).
LIVE_MAX_SECONDS = 60   # (live only) read/process the stream for up to this many seconds.
LIVE_EXPORT_MAX = 300   # (live only) max frames kept in memory for the .ply / .html export.
PREFETCH = True         # (file/folder only) load the next frame on a thread to overlap with GPU compute.
LONG_SIDE = 280         # resolution: the long side is resized to this many px. 504 is the model default, but
                        # the global-attention matrix grows with (window x patches)^2: on a 4 GB GPU use 280
                        # (or lower) so it fits in VRAM instead of spilling to shared system memory.
PATCH = 14              # patch size in px; each side is rounded to a multiple of this (model default 14).
DEVICE = "cuda"         # "cuda" for the GPU, "cpu" to force CPU.
CHECKPOINT = r"D:\v3R\V3R-solv3r\checkpoints\solv3r.safetensors"  # None = released weights (checkpoints/ if
                        # present, else Hugging Face), or a local .safetensors path / HF repo id.
DEPTH = True            # also predict per-frame depth (needed for the .ply / .html point clouds).
KEEP = 1.0              # fraction of points kept in the .ply (1.0 = all; most confident first).
SKY_MASK = False        # True drops the sky from the point cloud (outdoor; needs onnxruntime).
OUT_DIR = r"D:\v3R\V3R-solv3r\outputs\qstream"  # where stream.npz / stream.ply / stream.html are written.

# --- streaming window (solv3r.streaming.StreamConfig) — 4 GB-safe set; the paper's L8 (memory 64,
# recent 6, calls 8) needs a datacenter GPU. Held frames enter every call's global attention, which is
# quadratic in the window size, so keep MEMORY small on a small GPU. ---
ANCHORS = 8             # first frames, solved jointly once and then held fixed.
RECENT = 4              # most recent frames, kept being refined.
MEMORY = 8              # how many past frames (every MEMORY_STRIDE-th) are held in memory.
MEMORY_STRIDE = 8       # every MEMORY_STRIDE-th frame that leaves the recent window joins the memory.
CALLS = 4               # solver calls per frame.
ANCHOR_CALLS = 8        # solver calls for the one-time joint solve of the anchors.
REFRESH_TAU = 0.02      # above this relative change, a frame's calls update the whole window.
REFRESH_CHECK = 4       # every REFRESH_CHECK frames, measure the held frames' drift.
# --------------------------------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "solv3r"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from solv3r import Solv3R, StreamingSolver, list_images, load_images  # noqa: E402
from solv3r.export import write_ply  # noqa: E402
from solv3r.images import model_size  # noqa: E402
from solv3r.streaming import StreamConfig, StreamResult  # noqa: E402
from solv3r.viewer import save_stream_viewer  # noqa: E402
from solv3r.viewer.pack import stream_export, to_u8  # noqa: E402

# CLI overrides the first three configs.
INPUT = sys.argv[1] if len(sys.argv) > 1 else INPUT
STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else STRIDE
OUT_DIR = sys.argv[3] if len(sys.argv) > 3 else OUT_DIR
LIVE_MAX_SECONDS = int(sys.argv[4]) if len(sys.argv) > 4 else LIVE_MAX_SECONDS

LIVE = INPUT.lower().startswith(("rtmp://", "rtmps://", "http://", "https://"))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class LiveReader:
    """Read a live stream on a background thread into a small queue of the LATEST frames (stale frames
    are dropped), so the main loop stays live and processes as fast as the GPU allows."""

    def __init__(self, url, size, max_seconds, queue_size=8):
        import imageio_ffmpeg
        self.w, self.h = size[1], size[0]
        self.fbs = self.w * self.h * 3
        self.end = time.time() + max_seconds
        self.q = queue.Queue(maxsize=queue_size)
        self.proc = subprocess.Popen(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-rw_timeout", "5000000", "-i", url,
             "-vf", f"scale={self.w}:{self.h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while time.time() < self.end:
            buf = self.proc.stdout.read(self.fbs)
            if len(buf) < self.fbs:
                break
            arr = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
            frame = torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255
            try:
                self.q.put_nowait(frame)
            except queue.Full:  # drop the oldest so we keep the most recent (live)
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.q.put_nowait(frame)
                except queue.Full:
                    pass
        self.proc.terminate()

    def __iter__(self):
        return self

    def __next__(self):
        if time.time() > self.end and self.q.empty():
            raise StopIteration
        return self.q.get()


def live_frames(url, size, max_seconds):
    """Yield (3, h, w) float frames in [0, 1] from a live stream (latest frames, stale ones dropped)."""
    return LiveReader(url, size, max_seconds)


model = Solv3R.from_pretrained(CHECKPOINT, device=DEVICE)

if LIVE:
    paths, n = [], None
    # preserve the source aspect ratio (probing the stream) instead of a square frame: fewer patches,
    # and attention is quadratic in the patch count.
    try:
        import cv2
        cap = cv2.VideoCapture(INPUT)
        w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        cap.release()
        size = model_size(h0, w0, LONG_SIDE)
    except Exception:
        size = (max(PATCH, round(LONG_SIDE / PATCH) * PATCH),) * 2
    frame_iter = live_frames(INPUT, size, LIVE_MAX_SECONDS)
    stored = []  # (index, uint8 frame) kept for the .ply / .html export
else:
    paths = list_images(INPUT, stride=STRIDE, max_images=MAX_IMAGES)
    n = len(paths)
    if LONG_SIDE == 504:
        size = load_images(paths[:1]).shape[-2:]
    else:
        from PIL import Image, ImageOps
        with Image.open(paths[0]) as im0:
            w0, h0 = ImageOps.exif_transpose(im0).size
        size = model_size(h0, w0, LONG_SIDE)

    def load_one(p):
        return load_images([p], size)[0]

    if PREFETCH:
        q = queue.Queue(maxsize=3)

        def _worker():
            for p in paths:
                q.put(load_one(p))
            q.put(None)

        threading.Thread(target=_worker, daemon=True).start()
        frame_iter = iter(lambda: q.get(), None)
    else:
        frame_iter = (load_one(p) for p in paths)

cfg = StreamConfig(anchors=ANCHORS, recent=RECENT, memory=MEMORY, memory_stride=MEMORY_STRIDE,
                   calls=CALLS, anchor_calls=ANCHOR_CALLS, refresh_tau=REFRESH_TAU, refresh_check=REFRESH_CHECK)
solver = StreamingSolver(model, cfg, depth=DEPTH)

t0 = time.perf_counter()
i = 0
for frame in frame_iter:
    tp = time.perf_counter()
    answered = solver.push(frame)
    dt = (time.perf_counter() - tp) * 1000
    if LIVE and len(stored) < LIVE_EXPORT_MAX:
        stored.append((i, to_u8(frame[None])[0]))
    if answered or i % 10 == 0:
        log(f"frame {i + 1:4d}: push {dt:7.1f} ms  answered {answered}")
    i += 1
n = i
res = solver.result()
wall = time.perf_counter() - t0
log(f"streamed {n} frames in {wall:.2f}s ({wall / n * 1000:.1f} ms/frame wall, "
    f"{res.ms_per_frame:.1f} ms/frame solver), memory refreshed at {sum(res.refreshed)} frames")

os.makedirs(OUT_DIR, exist_ok=True)
prefix = os.path.join(OUT_DIR, "stream")
names = [os.path.basename(p) for p in paths] if not LIVE else [f"frame_{k:06d}" for k in range(n)]
np.savez_compressed(f"{prefix}.npz", c2w=res.c2w, c2w_exit=res.c2w_exit, K=res.intrinsics,
                    depth=res.depth.astype(np.float16), depth_conf=res.depth_conf.astype(np.float16),
                    names=np.array(names))


def part(sel):  # the stream's result for the frames sel
    return StreamResult(**{f: (getattr(res, f)[sel] if isinstance(getattr(res, f), np.ndarray) and len(getattr(res, f)) == n
                               else getattr(res, f)) for f in res.__dataclass_fields__})


if LIVE:
    sel = [k for k, _ in stored]
    frames_of = lambda s: torch.cat([dict(stored)[k][None] for k in s])  # noqa: E731
else:
    def frames_of(s):  # uint8 frames, read a chunk at a time (long videos)
        return torch.cat([to_u8(load_images([paths[i] for i in s[k:k + 64]], size)) for k in range(0, len(s), 64)])
    every = max(1, -(-n * size[0] * size[1] * KEEP // 20e6))
    sel = list(range(0, n, int(every)))

xyz, rgb = stream_export(frames_of(sel), part(sel), KEEP, SKY_MASK)
write_ply(f"{prefix}.ply", xyz, rgb)

every = max(1, -(-len(sel) // 1500))
vsel = sel[::int(every)]
save_stream_viewer(f"{prefix}.html", frames_of(vsel), part(vsel), [names[i] for i in vsel], sky=SKY_MASK,
                   per_frame=int(np.clip(1_000_000 / len(vsel), 300, 8000)),
                   title=os.path.basename(os.path.normpath(INPUT)) if not LIVE else "live",
                   stats=[f"{n} frames", f"{res.ms_per_frame:.0f} ms per frame"])

print(f"input: {INPUT}  ({n} frames, live={LIVE})")
print(f"c2w: {tuple(res.c2w.shape)}  K: {tuple(res.intrinsics.shape)}  depth: {tuple(res.depth.shape)}")
print(f"wrote: {prefix}.npz  {prefix}.ply  {prefix}.html")