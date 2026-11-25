"""Solv3R quick start on a drone video, with per-frame timing logged.

`list_images` reads a video file as its frames, so the README quick start runs on
samples/drone1.mp4 unchanged apart from a stride (378 frames is too many for a 4 GB GPU).

Writes <OUT_DIR>/solve.npz (cameras, intrinsics, depth, confidence), <OUT_DIR>/solve.ply (point
cloud) and <OUT_DIR>/solve.html (browser viewer). Logs encode / solve-call / decode timings and
the per-frame cost.

usage: python tests/qs.py [video|frames_folder] [stride] [out_dir] [calls]
"""
import os
import sys
import time

# --------------------------------------------------------------------------------------------------
# CONFIGS — every knob of the Solv3R quick start. Edit here, or override the first four on the CLI.
# --------------------------------------------------------------------------------------------------
INPUT = r"D:\v3R\V3R-solv3r\samples\ulab.mp4"  # a video file (.mp4/.mov/...) OR a folder of .jpg/.png
STRIDE = 100            # use every STRIDE-th frame (1 = all). Lower = more views, more VRAM/time.
MAX_IMAGES = None       # cap the number of views (None = no cap). Applied after STRIDE.
LONG_SIDE = 504         # resolution: the long side is resized to this many px (model default 504).
PATCH = 14              # patch size in px; each side is rounded to a multiple of this (model default 14).
CALLS = 16              # number of looped solver calls (more calls = tighter solve, slower).
ENCODE_CHUNK = 32       # images encoded per forward chunk (lower = less VRAM during encode).
DEVICE = "cuda"         # "cuda" for the GPU, "cpu" to force CPU.
CHECKPOINT = r"D:\v3R\V3R-solv3r\checkpoints\solv3r.safetensors"  # None = released weights (checkpoints/ if
                        # present, else Hugging Face), or a local .safetensors path / HF repo id. A local file
                        # avoids the "unauthenticated requests to the HF Hub" warning.
KEEP = 1.0              # fraction of points kept in the .ply (1.0 = all; most confident first).
SKY_MASK = True        # True drops the sky from the point cloud (outdoor; needs onnxruntime + download).
VIEW_KEEP = 1.0         # per-view point budget fraction (the viewer's per-view cap; 1.0 = all).
HTML_MAX_POINTS = 300000  # max points embedded in the .html viewer (larger clouds are subsampled).
OUT_DIR = r"D:\v3R\V3R-solv3r\outputs\qs"  # where solve.npz / solve.ply / solve.html are written.
# --------------------------------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "solv3r"))

from solv3r import Solv3R, cameras_from_pose_enc, list_images, load_images  # noqa: E402
from solv3r.export import save_solve, scene_points, write_html  # noqa: E402
from solv3r.images import model_size  # noqa: E402
from solv3r.model import autocast  # noqa: E402

# CLI overrides the first four configs.
INPUT = sys.argv[1] if len(sys.argv) > 1 else INPUT
STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else STRIDE
OUT_DIR = sys.argv[3] if len(sys.argv) > 3 else OUT_DIR
CALLS = int(sys.argv[4]) if len(sys.argv) > 4 else CALLS


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


model = Solv3R.from_pretrained(CHECKPOINT, device=DEVICE)  # the released weights
paths = list_images(INPUT, stride=STRIDE, max_images=MAX_IMAGES)
n = len(paths)

if LONG_SIDE == 504:
    images = load_images(paths)  # long side 504, sides multiples of 14, no crop
else:
    from PIL import Image, ImageOps
    with Image.open(paths[0]) as im0:
        w0, h0 = ImageOps.exif_transpose(im0).size
    images = load_images(paths, size=model_size(h0, w0, LONG_SIDE))  # custom resolution

t = time.perf_counter()
ctx = model.encode(images, chunk=ENCODE_CHUNK)  # the first image defines the world frame
enc = time.perf_counter() - t
log(f"encode {n} frames: {enc:.2f}s ({enc / n * 1000:.1f} ms/frame)")

z = model.start(ctx)
solve_total = 0.0
for call in range(1, CALLS + 1):
    t = time.perf_counter()
    with autocast():
        z = model.step(z, ctx)
    dt = time.perf_counter() - t
    solve_total += dt
    log(f"solve call {call:2d}/{CALLS}: {dt:.2f}s ({dt / n * 1000:.1f} ms/frame)")

t = time.perf_counter()
pred = model.decode(z, ctx)  # depth, depth_conf, rays, world_points, pose_enc
dec = time.perf_counter() - t
log(f"decode {n} frames: {dec:.2f}s ({dec / n * 1000:.1f} ms/frame)")

c2w, K = cameras_from_pose_enc(pred["pose_enc"], (ctx.H, ctx.W))

os.makedirs(OUT_DIR, exist_ok=True)
prefix = os.path.join(OUT_DIR, "solve")
names = [os.path.basename(p) for p in paths]
save_solve(prefix, images, pred, names, keep=KEEP, sky=SKY_MASK, html=False, view_keep=VIEW_KEEP)
xyz, rgb = scene_points(pred["world_points"].float().cpu().numpy(),
                        pred["depth_conf"].float().cpu().numpy(),
                        pred["depth"][..., 0].float().cpu().numpy(),
                        images, KEEP, SKY_MASK, VIEW_KEEP)
write_html(prefix + ".html", xyz, rgb, c2w.numpy(), max_points=HTML_MAX_POINTS,
           title=os.path.basename(os.path.normpath(INPUT)))

total = enc + solve_total + dec
log(f"total {total:.2f}s for {n} frames: {total / n * 1000:.1f} ms/frame "
    f"(encode {enc / n * 1000:.1f} + solve {solve_total / n * 1000:.1f} + decode {dec / n * 1000:.1f} ms/frame)")
print(f"input: {INPUT}  ({n} views)")
print(f"images: {tuple(images.shape)}  cameras: {tuple(c2w.shape)}  K: {tuple(K.shape)}")
print(f"depth: {tuple(pred['depth'].shape)}  world_points: {tuple(pred['world_points'].shape)}")
print(f"wrote: {prefix}.npz  {prefix}.ply  {prefix}.html")