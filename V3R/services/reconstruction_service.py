import bpy


def start(settings):
    """Start realtime reconstruction in a non-blocking manner.

    Heavy work should be offloaded to a background thread or scheduled
    via bpy.app.timers so the Blender UI stays responsive.
    """
    # Placeholder for future reconstruction pipeline.
    pass


def stop():
    """Stop realtime reconstruction and release resources."""
    pass
