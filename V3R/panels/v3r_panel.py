"""V3R sidebar panel: source, Solv3R settings, live status and run controls."""

from __future__ import annotations

from bpy.types import Panel


class V3R_PT_MainPanel(Panel):
    bl_label = "V3R"
    bl_idname = "V3R_PT_main_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "V3R"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.v3r_settings

        col = layout.column(align=True)
        col.prop(settings, "source_type", expand=True)
        if settings.source_type == "video":
            row = col.row(align=True)
            row.operator("v3r.select_video", text="Open Video", icon="FILE_FOLDER")
            row.prop(settings, "video_path", text="")
        else:
            col.prop(settings, "stream_link", text="Stream URL")

        col.separator()
        col.prop(settings, "output_dir")
        row = col.row(align=True)
        row.operator("v3r.open_output", text="Open Output", icon="FILE_PARENT")
        row.operator("v3r.clear_cloud", text="Clear Cloud", icon="TRASH")

        col.separator()
        col.prop(settings, "device", text="Device")
        col.prop(settings, "preset", text="Preset")
        col.prop(settings, "image_long_side", text="Resolution")
        row = col.row(align=True)
        row.prop(settings, "sample_fps", text="Sample FPS")
        row.prop(settings, "max_frames", text="Max Frames")

        col.separator()
        if settings.is_running:
            row = col.row()
            row.alert = True
            row.operator("v3r.toggle_reconstruction", text="Stop", icon="PAUSE")
        else:
            col.operator("v3r.toggle_reconstruction", text="Start Reconstruction", icon="PLAY")

        box = layout.box()
        box.label(text=settings.status_text or "Idle", icon="INFO")
        if settings.device_label:
            box.label(text=f"Device: {settings.device_label} · {settings.model_label}")
        row = box.row(align=True)
        row.label(text=f"Frames: {settings.frames_read}")
        row.label(text=f"Points: {settings.points_total}")
        row.label(text=f"{settings.ms_avg:.0f} ms/frame")
        if settings.depth_label:
            box.label(text=f"Latest depth: {settings.depth_label}", icon="IMAGE_DATA")
        if settings.error_text:
            error = box.row()
            error.alert = True
            error.label(text=settings.error_text, icon="ERROR")

        header, body = layout.panel("V3R_PT_advanced", default_closed=True)
        header.label(text="Advanced")
        if body:
            body.prop(settings, "worker_python_path")
            body.prop(settings, "worker_script_path")
            body.prop(settings, "checkpoint_path")
            body.prop(settings, "stream_lost_timeout_s")
            body.prop(settings, "depth_enabled")
            body.prop(settings, "point_stride")
            body.prop(settings, "point_budget")
            body.prop(settings, "point_radius")
            col = body.column(align=True)
            col.label(text="Solv3R streaming")
            col.prop(settings, "anchors")
            col.prop(settings, "recent")
            col.prop(settings, "memory")
            col.prop(settings, "memory_stride")
            col.prop(settings, "calls")
            col.prop(settings, "anchor_calls")
            col.prop(settings, "refresh_tau")
            col.prop(settings, "refresh_check")
