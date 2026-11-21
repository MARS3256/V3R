"""Render the reconstructed point cloud in EEVEE after a short Solv3R run."""

from __future__ import annotations

import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import bpy  # noqa: E402
import mathutils  # noqa: E402
import numpy as np  # noqa: E402

import V3R  # noqa: E402
from V3R.services import reconstruction_service  # noqa: E402


def main() -> None:
    out_dir = os.path.join(ROOT, "samples", "outputs", "blender_render")
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
    settings.max_frames = 8
    settings.point_budget = 100000
    settings.point_radius = 0.15

    reconstruction_service.start(settings)
    for _ in range(400):
        time.sleep(0.2)
        reconstruction_service._poll()
        if not reconstruction_service.is_running():
            break
    reconstruction_service.stop()

    obj = bpy.data.objects.get("V3R_PointCloud")
    n = len(obj.data.points)
    co = np.empty(n * 3, np.float32)
    obj.data.points.foreach_get("co", co)
    xyz = co.reshape(-1, 3)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    center = (lo + hi) / 2.0
    size = float(np.linalg.norm(hi - lo))
    print("POINTS", n, "SIZE", size)

    bpy.ops.object.camera_add(location=center + mathutils.Vector((size, -size, size * 0.7)))
    cam = bpy.context.object
    cam.rotation_euler = (mathutils.Vector(center) - cam.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    bpy.context.scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.02, 0.02, 0.02, 1)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 16
    scene.render.resolution_x = 960
    scene.render.resolution_y = 640
    scene.render.filepath = os.path.join(out_dir, "blender_points_eevee.png")
    bpy.ops.render.render(write_still=True)
    print("RENDER_DONE", scene.render.filepath)


if __name__ == "__main__":
    main()
