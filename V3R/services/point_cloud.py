"""Blender point-cloud object helpers for V3R.

Uses Blender's native Point Cloud object: one point per sample with a per-point radius and a
FLOAT_COLOR "Col" attribute. Cycles and EEVEE render it natively (EEVEE as low-poly sphere
instances), so millions of points do not become millions of meshes.

This module uses bpy and numpy and never imports torch or the worker package.
"""

from __future__ import annotations

import bpy
import numpy as np

OBJECT_NAME = "V3R_PointCloud"
MATERIAL_NAME = "V3R_PointMaterial"
COLOUR_ATTRIBUTE = "Col"
RADIUS_ATTRIBUTE = "radius"
DEFAULT_RADIUS = 0.1


def apply_points(xyz: np.ndarray, rgb: np.ndarray, radius: float = DEFAULT_RADIUS) -> bpy.types.Object:
    """Write xyz (N, 3) float and rgb (N, 3) uint8 into the native point cloud."""
    xyz = np.asarray(xyz, np.float32)
    rgb = np.asarray(rgb, np.uint8)
    n = len(xyz)
    obj = _ensure_object()
    cloud = obj.data
    cloud.resize(n)
    if n:
        cloud.points.foreach_set("co", xyz.ravel())
        radius_data = cloud.attributes.get(RADIUS_ATTRIBUTE)
        if radius_data is None:
            radius_data = cloud.attributes.new(RADIUS_ATTRIBUTE, "FLOAT", "POINT")
        radius_data.data.foreach_set("value", np.full(n, radius, np.float32))
        colour = cloud.attributes.get(COLOUR_ATTRIBUTE)
        if colour is None:
            colour = cloud.attributes.new(COLOUR_ATTRIBUTE, "FLOAT_COLOR", "POINT")
        rgba = np.ones((n, 4), np.float32)
        rgba[:, :3] = rgb.astype(np.float32) / 255.0
        colour.data.foreach_set("color", rgba.ravel())
    _ensure_material(cloud)
    cloud.update_tag()
    return obj


def clear_points() -> None:
    """Empty the point cloud without touching the worker."""
    obj = bpy.data.objects.get(OBJECT_NAME)
    if obj is not None and obj.type == "POINTCLOUD":
        obj.data.resize(0)
        obj.data.update_tag()


def _ensure_object() -> bpy.types.Object:
    """Return the V3R point-cloud object, creating it in the scene on first use."""
    obj = bpy.data.objects.get(OBJECT_NAME)
    if obj is not None:
        if obj.type != "POINTCLOUD":
            raise RuntimeError(f"object {OBJECT_NAME} exists but is {obj.type}, not POINTCLOUD")
        return obj
    cloud = bpy.data.pointclouds.new(OBJECT_NAME)
    obj = bpy.data.objects.new(OBJECT_NAME, cloud)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _ensure_material(cloud: bpy.types.PointCloud) -> None:
    """Add the vertex-colour emission material to the point cloud once."""
    if cloud.materials.get(MATERIAL_NAME) is not None:
        return
    material = bpy.data.materials.get(MATERIAL_NAME)
    if material is None:
        material = bpy.data.materials.new(MATERIAL_NAME)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        nodes.clear()
        colour = nodes.new("ShaderNodeAttribute")
        colour.attribute_name = COLOUR_ATTRIBUTE
        colour.attribute_type = "GEOMETRY"
        emission = nodes.new("ShaderNodeEmission")
        output = nodes.new("ShaderNodeOutputMaterial")
        material.node_tree.links.new(colour.outputs["Color"], emission.inputs["Color"])
        material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    cloud.materials.append(material)
