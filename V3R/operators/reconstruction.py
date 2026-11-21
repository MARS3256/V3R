"""V3R operators: start/stop the worker, open the output folder and clear the cloud."""

from __future__ import annotations

import bpy
from bpy.types import Operator

from ..services import reconstruction_service
from ..services.point_cloud import clear_points


class V3R_OT_ToggleReconstruction(Operator):
    """Start or stop the Solv3R reconstruction worker"""

    bl_idname = "v3r.toggle_reconstruction"
    bl_label = "Start / Stop V3R"
    bl_description = "Start or stop the Solv3R reconstruction worker (non-blocking)"
    bl_options = {"REGISTER"}

    def execute(self, context) -> set[str]:
        settings = context.scene.v3r_settings
        if settings.is_running:
            reconstruction_service.stop()
            settings.is_running = False
            self.report({"INFO"}, "Stopped V3R reconstruction")
            return {"FINISHED"}
        try:
            reconstruction_service.start(settings)
        except Exception as exc:
            settings.is_running = False
            settings.error_text = str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, "Started V3R reconstruction")
        return {"FINISHED"}


class V3R_OT_OpenOutput(Operator):
    """Open the V3R run folder in the system file browser"""

    bl_idname = "v3r.open_output"
    bl_label = "Open Output Folder"
    bl_description = "Open the folder that holds depth maps, points and the report"
    bl_options = {"REGISTER"}

    def execute(self, context) -> set[str]:
        settings = context.scene.v3r_settings
        path = reconstruction_service.resolve_output_dir(settings)
        if path is None:
            self.report({"ERROR"}, "output folder is not set")
            return {"CANCELLED"}
        bpy.ops.wm.path_open(filepath=str(path))
        return {"FINISHED"}


class V3R_OT_ClearCloud(Operator):
    """Remove all points from the V3R point cloud"""

    bl_idname = "v3r.clear_cloud"
    bl_label = "Clear Point Cloud"
    bl_description = "Empty the V3R point cloud without touching the worker"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context) -> set[str]:
        clear_points()
        return {"FINISHED"}
