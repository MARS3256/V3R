bl_info = {
    "name": "V3R",
    "author": "V3R",
    "version": (1, 0, 0),
    "blender": (5, 2, 0),
    "location": "View3D > Sidebar > V3R",
    "description": "Realtime 3D reconstruction from video and live streams",
    "warning": "",
    "doc_url": "",
    "category": "3D View",
}

import bpy
from bpy.types import Scene

from .properties.v3r_settings import V3RSettings
from .operators.select_video import V3R_OT_SelectVideo
from .operators.reconstruction import V3R_OT_ToggleReconstruction
from .panels.v3r_panel import V3R_PT_MainPanel

_CLASSES = (
    V3RSettings,
    V3R_OT_SelectVideo,
    V3R_OT_ToggleReconstruction,
    V3R_PT_MainPanel,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    Scene.v3r_settings = bpy.props.PointerProperty(type=V3RSettings)


def unregister():
    del Scene.v3r_settings
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
