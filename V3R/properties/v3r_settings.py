"""V3R add-on settings stored on the scene.

Every Solv3R streaming parameter is exposed here. The worker receives them in its JSON config;
the Blender side never imports torch or the worker package.
"""

from __future__ import annotations

from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from bpy.types import PropertyGroup

_PRESETS = {
    "S4": (4, 4, 16, 16, 4, 16),
    "L4": (8, 6, 64, 16, 4, 16),
    "L8": (8, 6, 64, 16, 8, 16),
}


def _apply_preset(self: PropertyGroup, context: object) -> None:
    """Copy the selected Solv3R preset into the advanced stream parameters."""
    anchors, recent, memory, stride, calls, anchor_calls = _PRESETS[self.preset]
    self.anchors = anchors
    self.recent = recent
    self.memory = memory
    self.memory_stride = stride
    self.calls = calls
    self.anchor_calls = anchor_calls


class V3RSettings(PropertyGroup):
    """All V3R settings, including the Solv3R streaming parameters."""

    source_type: EnumProperty(
        name="Source",
        description="Read a video file or a live stream",
        items=(
            ("video", "Video File", "Read frames from a video file"),
            ("stream", "Live Stream", "Read frames from an RTSP/RTMP/UDP/HTTP stream"),
        ),
        default="video",
    )
    video_path: StringProperty(
        name="Video Path",
        description="Video file used for reconstruction",
        default="",
        subtype="FILE_PATH",
    )
    stream_link: StringProperty(
        name="Stream Link",
        description="RTSP, RTMP, UDP, SRT or HTTP(S) stream URL (OBS and vdo.ninja via OBS output)",
        default="",
    )
    output_dir: StringProperty(
        name="Output Folder",
        description="Run folder for depth maps, points and report",
        default="samples/outputs",
        subtype="DIR_PATH",
    )
    checkpoint_path: StringProperty(
        name="Model Weights",
        description="Local Solv3R .safetensors checkpoint",
        default="models/solv3r.safetensors",
        subtype="FILE_PATH",
    )
    worker_python_path: StringProperty(
        name="Worker Python",
        description=(
            "Python interpreter of the worker venv; empty uses .venv-worker "
            "(bin/python on Linux, Scripts/python.exe on Windows)"
        ),
        default="",
        subtype="FILE_PATH",
    )
    worker_script_path: StringProperty(
        name="Worker Script",
        description="Path to worker/app.py",
        default="worker/app.py",
        subtype="FILE_PATH",
    )

    device: EnumProperty(
        name="Device",
        description="Compute backend; an unavailable backend is an error, never a fallback",
        items=(
            ("cpu", "CPU", "Run on the processor"),
            ("cuda", "CUDA / ROCm", "Run on the GPU when the installed PyTorch supports it"),
        ),
        default="cpu",
    )
    preset: EnumProperty(
        name="Preset",
        description="Solv3R streaming preset",
        items=(
            ("S4", "S4 (fast)", "4 anchors, 4 recent, 4 calls"),
            ("L4", "L4", "8 anchors, 6 recent, 4 calls"),
            ("L8", "L8 (accurate)", "8 anchors, 6 recent, 8 calls"),
        ),
        default="S4",
        update=_apply_preset,
    )
    image_long_side: EnumProperty(
        name="Image Long Side",
        description="Model input resolution; 252 is faster, 504 is sharper",
        items=(("252", "252 px", "Fast, lower detail"), ("504", "504 px", "Sharper, slower")),
        default="252",
    )
    sample_fps: FloatProperty(
        name="Sample FPS",
        description="Frames per second fed to Solv3R; 0 uses every frame",
        default=2.0,
        min=0.0,
        max=120.0,
        soft_max=30.0,
    )
    max_frames: IntProperty(
        name="Max Frames",
        description="Stop after this many sampled frames; 0 means no limit",
        default=0,
        min=0,
        soft_max=100000,
    )
    point_stride: IntProperty(
        name="Point Stride",
        description="Pixel stride when converting depth to points",
        default=3,
        min=1,
        max=32,
    )
    point_budget: IntProperty(
        name="Point Budget",
        description="Maximum points kept in the Blender point cloud",
        default=300000,
        min=1000,
        max=5000000,
    )
    point_radius: FloatProperty(
        name="Point Radius",
        description="Rendered radius of each point, in metres",
        default=0.1,
        min=0.001,
        max=10.0,
        soft_max=2.0,
    )
    depth_enabled: BoolProperty(
        name="Write Depth Maps",
        description="Save depth NPZ and colourised PNG for every answered frame",
        default=True,
    )
    stream_lost_timeout_s: FloatProperty(
        name="Stream Lost Timeout",
        description="Seconds without a frame before a live stream is declared lost",
        default=10.0,
        min=1.0,
        max=300.0,
    )

    anchors: IntProperty(name="Anchors", default=4, min=1, max=64)
    recent: IntProperty(name="Recent Frames", default=4, min=1, max=64)
    memory: IntProperty(name="Memory Frames", default=16, min=0, max=512)
    memory_stride: IntProperty(name="Memory Stride", default=16, min=1, max=256)
    calls: IntProperty(name="Solver Calls", default=4, min=1, max=64)
    anchor_calls: IntProperty(name="Anchor Calls", default=16, min=1, max=64)
    refresh_tau: FloatProperty(name="Refresh Tau", default=0.02, min=0.0, max=1.0)
    refresh_check: IntProperty(name="Refresh Check", default=4, min=1, max=64)

    is_running: BoolProperty(name="Running", default=False)
    status_text: StringProperty(name="Status", default="")
    error_text: StringProperty(name="Error", default="")
    device_label: StringProperty(name="Device", default="")
    model_label: StringProperty(name="Model", default="")
    depth_label: StringProperty(name="Latest Depth", default="")
    frames_read: IntProperty(name="Frames", default=0)
    points_total: IntProperty(name="Points", default=0)
    ms_avg: FloatProperty(name="Average ms", default=0.0)
