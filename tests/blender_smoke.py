"""Blender background smoke test for the V3R add-on and Solv3R worker.

Run with:
  blender -b --python tests/blender_smoke.py
"""

from __future__ import annotations

import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import bpy  # noqa: E402

import V3R  # noqa: E402
from V3R.services import reconstruction_service  # noqa: E402


def main() -> None:
    out_dir = os.path.join(ROOT, "samples", "outputs", "blender_test")
    shutil.rmtree(out_dir, ignore_errors=True)
    V3R.register()
    settings = bpy.context.scene.v3r_settings
    settings.source_type = "video"
    settings.video_path = os.path.join(ROOT, "samples", "drone1.mp4")
    settings.output_dir = out_dir
    settings.worker_python_path = os.path.join(ROOT, ".venv-worker", "bin", "python")
    settings.worker_script_path = os.path.join(ROOT, "worker", "app.py")
    settings.checkpoint_path = os.path.join(ROOT, "models", "solv3r.safetensors")
    settings.device = "cpu"
    settings.preset = "S4"
    settings.image_long_side = "252"
    settings.sample_fps = 2.0
    settings.max_frames = 6
    settings.point_budget = 100000

    reconstruction_service.start(settings)
    for _ in range(300):
        time.sleep(0.2)
        reconstruction_service._poll()
        if not reconstruction_service.is_running():
            break
    reconstruction_service.stop()

    obj = bpy.data.objects.get("V3R_PointCloud")
    print("SMOKE points", len(obj.data.points) if obj else -1)
    print("SMOKE frames", settings.frames_read)
    print("SMOKE status", settings.status_text)
    print("SMOKE error", settings.error_text)


if __name__ == "__main__":
    main()
