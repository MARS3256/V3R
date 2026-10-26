import bpy
from bpy.props import (
    StringProperty,
    BoolProperty,
    IntProperty,
    FloatProperty,
)
from bpy.types import PropertyGroup


class V3RSettings(PropertyGroup):
    """V3R addon settings stored on the scene."""

    video_path: StringProperty(
        name="Video Path",
        description="Path to the video file used for 3D reconstruction",
        default="",
        subtype='FILE_PATH',
    )

    stream_link: StringProperty(
        name="Stream Link",
        description="Live video stream URL (e.g. http://localhost:8080/stream)",
        default="",
    )

    is_running: BoolProperty(
        name="Running",
        description="Whether realtime 3D reconstruction is currently running",
        default=False,
    )

    debug_mode: BoolProperty(
        name="Debug Mode",
        description="Enable verbose debug logging and diagnostics",
        default=False,
    )

    camera_fov: FloatProperty(
        name="Camera FOV",
        description="Camera field of view in degrees, used for reconstruction calculations",
        default=60.0,
        min=10.0,
        max=150.0,
        soft_min=10.0,
        soft_max=150.0,
    )

    stm_frames: IntProperty(
        name="STM Frames",
        description="Short Term Memory: number of frames providing spatial data for a sequence",
        default=8,
        min=1,
        max=1024,
    )
