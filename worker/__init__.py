"""V3R worker package: Solv3R inference, frame ingest and run outputs.

The worker is a separate process. It never imports bpy. Blender talks to it through a JSON
config file, JSON status lines on stdout, and the run folder under samples/outputs.
"""
