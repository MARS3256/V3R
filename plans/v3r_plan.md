# V3R Blender Plugin Plan

## Goal
Create Blender 5.2 LTS extension "V3R" with N-panel UI, modular hierarchical folder layout separating business logic from Blender UI logic.

## Research
- Blender 5.2 uses Extensions platform (blender_manifest.toml) + legacy bl_info dual-mode.
- Multi-file package structure recommended.
- N-panel = VIEW_3D / UI region / bl_category.
- File browser via StringProperty subtype FILE_PATH or operator fileselect_add.
- Non-blocking via bpy.app.timers / threading; for now notification + toggle.
- Collapsible settings via layout.panel() or DEFAULT_CLOSED sub-panel.

## Folder Layout
V3R/
  __init__.py
  blender_manifest.toml
  properties/   (data models)
  operators/     (Blender operators / UI actions)
  panels/        (Blender panels)
  services/      (business logic, no bpy UI)
  utils/

## UI Spec
Column:
  Video Path: (file opener button)
  Stream Link: (string)
  Start/Stop button (blue/red toggle, notification)
Collapsible settings:
  Debug mode (bool)
  Camera FOV slider 10-150 deg
  STM Frames int

All items tooltips.

## Build
Use `blender --command extension build`.
