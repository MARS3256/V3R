import bpy
from bpy.types import Panel


class V3R_PT_MainPanel(Panel):
    bl_label = "V3R"
    bl_idname = "V3R_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "V3R"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.v3r_settings

        col = layout.column(align=True)

        # Video Path
        row = col.row(align=True)
        row.label(text="Video Path:")
        row.operator("v3r.select_video", text="Open File", icon='FILE_FOLDER')
        if settings.video_path:
            col.label(text=settings.video_path, icon='FILE')

        # Stream Link
        col.prop(settings, "stream_link", text="Stream Link")

        # Start / Stop
        row = col.row()
        if settings.is_running:
            row.alert = True
            row.operator("v3r.toggle_reconstruction", text="Stop", icon='PAUSE')
        else:
            row.operator("v3r.toggle_reconstruction", text="Start", icon='PLAY')

        # Collapsible settings area
        header, body = layout.panel("V3R_PT_settings_panel", default_closed=True)
        header.label(text="Settings")
        if body:
            body.prop(settings, "debug_mode", text="Debug Mode")
            body.prop(settings, "camera_fov", text="Camera FOV", slider=True)
            body.prop(settings, "stm_frames", text="STM Frames")
