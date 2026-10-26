import bpy
from bpy.types import Operator
from ..services import reconstruction_service


class V3R_OT_ToggleReconstruction(Operator):
    """Start or stop the realtime 3D reconstruction process"""

    bl_idname = "v3r.toggle_reconstruction"
    bl_label = "Toggle Reconstruction"
    bl_description = "Start or stop realtime 3D reconstruction (non-blocking)"
    bl_options = {'REGISTER'}

    def execute(self, context):
        settings = context.scene.v3r_settings
        if not settings.is_running:
            reconstruction_service.start(settings)
            settings.is_running = True
            self.report({'INFO'}, "Started 3D reconstruction")
        else:
            reconstruction_service.stop()
            settings.is_running = False
            self.report({'INFO'}, "Stopped 3D reconstruction")
        return {'FINISHED'}
