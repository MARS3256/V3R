import bpy
from bpy.types import Scene

from .operators.reconstruction import (
    V3R_OT_ClearCloud,
    V3R_OT_OpenOutput,
    V3R_OT_ToggleReconstruction,
)
from .operators.select_video import V3R_OT_SelectVideo
from .panels.v3r_panel import V3R_PT_MainPanel
from .properties.v3r_settings import V3RSettings

bl_info = {
    "name": "V3R",
    "author": "V3R",
    "version": (2, 0, 0),
    "blender": (5, 2, 0),
    "location": "View3D > Sidebar > V3R",
    "description": "Realtime Solv3R reconstruction from video and live streams",
    "warning": "",
    "doc_url": "",
    "category": "3D View",
}

_CLASSES = (
    V3RSettings,
    V3R_OT_SelectVideo,
    V3R_OT_ToggleReconstruction,
    V3R_OT_OpenOutput,
    V3R_OT_ClearCloud,
    V3R_PT_MainPanel,
)


def register() -> None:
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    Scene.v3r_settings = bpy.props.PointerProperty(type=V3RSettings)


def unregister() -> None:
    del Scene.v3r_settings
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
