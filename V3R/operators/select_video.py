import bpy
import os
from bpy.props import StringProperty
from bpy_extras.io_utils import ImportHelper
from bpy.types import Operator


class V3R_OT_SelectVideo(Operator, ImportHelper):
    """Open Blender's native file browser to select a video file"""

    bl_idname = "v3r.select_video"
    bl_label = "Select Video"
    bl_description = "Open Blender's file browser to choose a video file for reconstruction"
    bl_options = {'REGISTER'}

    filename_ext = ""
    filter_glob: StringProperty(
        default="*.mp4;*.avi;*.mkv;*.mov;*.webm;*.m4v;*.mpg;*.mpeg",
        options={'HIDDEN'},
        name="File Filter",
        description="Supported video file formats",
    )

    filepath: StringProperty(
        name="File Path",
        description="Path to the selected video file",
        subtype='FILE_PATH',
        default="",
    )

    def execute(self, context):
        settings = context.scene.v3r_settings
        settings.video_path = self.filepath
        name = os.path.basename(self.filepath) if self.filepath else ""
        self.report({'INFO'}, f"Video selected: {name}")
        return {'FINISHED'}
