"""Pure WGS84 and camera geometry: lat/lon/alt to local ENU, camera rays and ground hits.

This module does NOT read files, use a clock, or import bpy. All functions are numpy in,
numpy out (rule R14) and are the shared geometry contract for depth, scale fit and
back-projection.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from core.dataclasses import FlightSample

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3
_BODY_AT_REST = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
_MOUNT_AT_REST = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def enu_from_llh(origin: FlightSample, lat: float, lon: float, alt: float) -> tuple[float, float, float]:
    """Return (east, north, up) metres of a WGS84 point relative to the origin fix.

    Uses a local equirectangular approximation about the origin latitude, accurate to
    millimetres over a few kilometres. The origin is the first accepted telemetry fix.
    """
    lat0 = math.radians(origin.lat_deg)
    sin_lat0 = math.sin(lat0)
    dlat = math.radians(lat - origin.lat_deg)
    dlon = math.radians(lon - origin.lon_deg)
    prime_vertical = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat0 * sin_lat0)
    meridian = WGS84_A * (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * sin_lat0 * sin_lat0) ** 1.5
    east = dlon * prime_vertical * math.cos(lat0)
    north = dlat * meridian
    up = alt - origin.alt_m
    return east, north, up


def camera_rotation(
    sample: FlightSample,
    mount_tilt_deg: float,
    roll_offset_deg: float,
    yaw_offset_deg: float,
) -> NDArray[np.float64]:
    """Return the 3x3 camera-to-ENU rotation for one flight sample.

    World axes are X east, Y north, Z up. Camera +Z is the optical axis, +X image right,
    +Y image down. Body x is forward, y right, z down. The rotation is the body
    roll/pitch/yaw composition times the fixed mount tilt forward from nadir.
    """
    yaw = math.radians(sample.yaw_deg + yaw_offset_deg)
    pitch = math.radians(sample.pitch_deg)
    roll = math.radians(sample.roll_deg + roll_offset_deg)
    tilt = math.radians(mount_tilt_deg)
    body = _BODY_AT_REST @ _rot_x(roll) @ _rot_y(pitch) @ _rot_z(yaw)
    return body @ _rot_y(tilt) @ _MOUNT_AT_REST


def build_rays(width: int, height: int, fov_x_deg: float) -> NDArray[np.float32]:
    """Return normalized camera-ray directions with shape (3, height, width).

    Pixel centres are at (u + 0.5, v + 0.5); fx = (width / 2) / tan(fov_x / 2) with square
    pixels and the principal point at the image centre. Camera +Z is the optical axis.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    focal = (width / 2.0) / math.tan(math.radians(fov_x_deg) / 2.0)
    x = (np.arange(width, dtype=np.float64) + 0.5 - width / 2.0) / focal
    y = (np.arange(height, dtype=np.float64) + 0.5 - height / 2.0) / focal
    grid_x, grid_y = np.meshgrid(x, y)
    rays = np.stack((grid_x, grid_y, np.ones_like(grid_x)), axis=0)
    return (rays / np.linalg.norm(rays, axis=0, keepdims=True)).astype(np.float32)


def ray_ground_distance(
    dir_cam: NDArray[np.float32],
    rotation: NDArray[np.float64],
    cam_pos_enu: Sequence[float],
    ground_z: float,
) -> NDArray[np.float64]:
    """Return per-pixel slant distance to the plane z = ground_z, or inf when it never hits.

    dir_cam has shape (3, height, width) and is normalized in camera coordinates; rotation
    maps camera directions to ENU. Rays pointing up or hitting behind the camera return inf.
    """
    world = np.einsum("ij,jhw->ihw", rotation, dir_cam)
    dz = world[2]
    distance = np.full(dz.shape, np.inf, dtype=np.float64)
    np.divide(ground_z - float(cam_pos_enu[2]), dz, out=distance, where=dz < 0.0)
    distance[distance < 0.0] = np.inf
    return distance


def _rot_x(angle: float) -> NDArray[np.float64]:
    """Return a rotation matrix about the X axis."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(angle: float) -> NDArray[np.float64]:
    """Return a rotation matrix about the Y axis."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(angle: float) -> NDArray[np.float64]:
    """Return a rotation matrix about the Z axis."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
