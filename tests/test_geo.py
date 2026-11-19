"""Analytic geometry tests for the shared core/geo.py contract."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.dataclasses import FlightSample
from core.geo import build_rays, camera_rotation, enu_from_llh, ray_ground_distance

ORIGIN = FlightSample(0.0, 23.8, 90.4, 50.0, 0.0, 0.0, 0.0)


def _meridian(lat_deg: float) -> float:
    """Return the WGS84 meridian radius of curvature at a latitude."""
    e2 = 6.69437999014e-3
    a = 6378137.0
    phi = math.radians(lat_deg)
    return a * (1.0 - e2) / (1.0 - e2 * math.sin(phi) ** 2) ** 1.5


def _prime_vertical(lat_deg: float) -> float:
    """Return the WGS84 prime-vertical radius of curvature at a latitude."""
    e2 = 6.69437999014e-3
    a = 6378137.0
    phi = math.radians(lat_deg)
    return a / math.sqrt(1.0 - e2 * math.sin(phi) ** 2)


def test_enu_north_offset_is_100_metres() -> None:
    """A point 100 m north of the origin maps to (0, 100, 0)."""
    north_lat = math.degrees(math.radians(ORIGIN.lat_deg) + 100.0 / _meridian(ORIGIN.lat_deg))
    east, north, up = enu_from_llh(ORIGIN, north_lat, ORIGIN.lon_deg, ORIGIN.alt_m)
    assert east == pytest.approx(0.0, abs=1e-6)
    assert north == pytest.approx(100.0, abs=1e-6)
    assert up == pytest.approx(0.0, abs=1e-6)


def test_enu_east_offset_is_100_metres() -> None:
    """A point 100 m east of the origin maps to (100, 0, 0)."""
    radius = _prime_vertical(ORIGIN.lat_deg) * math.cos(math.radians(ORIGIN.lat_deg))
    east_lon = ORIGIN.lon_deg + math.degrees(100.0 / radius)
    east, north, up = enu_from_llh(ORIGIN, ORIGIN.lat_deg, east_lon, ORIGIN.alt_m)
    assert east == pytest.approx(100.0, abs=1e-6)
    assert north == pytest.approx(0.0, abs=1e-6)
    assert up == pytest.approx(0.0, abs=1e-6)


def test_enu_up_is_altitude_difference() -> None:
    """Up is the plain altitude difference from the origin."""
    _, _, up = enu_from_llh(ORIGIN, ORIGIN.lat_deg, ORIGIN.lon_deg, ORIGIN.alt_m + 12.5)
    assert up == pytest.approx(12.5, abs=1e-9)


def test_camera_forward_axis_with_mount_tilt() -> None:
    """A level body with 15 deg mount tilt maps camera +Z to north and down."""
    rotation = camera_rotation(ORIGIN, 15.0, 0.0, 0.0)
    forward = rotation @ np.array([0.0, 0.0, 1.0])
    expected = np.array([0.0, math.sin(math.radians(15.0)), -math.cos(math.radians(15.0))])
    assert np.allclose(forward, expected, atol=1e-12)


def test_camera_rotation_is_a_rotation() -> None:
    """camera_rotation returns an orthonormal matrix with determinant +1."""
    rotation = camera_rotation(ORIGIN, 15.0, 2.0, -3.0)
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12)


def test_build_rays_centre_pixel_points_forward() -> None:
    """The centre ray of an odd-sized image is exactly camera +Z and rays are normalized."""
    rays = build_rays(5, 5, 90.0)
    assert rays.shape == (3, 5, 5)
    assert np.allclose(rays[:, 2, 2], [0.0, 0.0, 1.0], atol=1e-6)
    assert np.allclose(np.linalg.norm(rays, axis=0), 1.0, atol=1e-6)


def test_build_rays_rejects_empty_image() -> None:
    """A zero-sized image is a caller error, not an empty ray grid."""
    with pytest.raises(ValueError):
        build_rays(0, 5, 90.0)


def test_ground_distance_for_nadir_camera() -> None:
    """A nadir camera 100 m up hits the ground at 100 m on the centre ray."""
    rotation = camera_rotation(ORIGIN, 0.0, 0.0, 0.0)
    rays = build_rays(5, 5, 90.0)
    distance = ray_ground_distance(rays, rotation, (0.0, 0.0, 100.0), 0.0)
    assert distance[2, 2] == pytest.approx(100.0, rel=1e-9)


def test_ground_distance_grows_off_axis() -> None:
    """Off-axis rays travel farther than the centre ray."""
    rotation = camera_rotation(ORIGIN, 0.0, 0.0, 0.0)
    rays = build_rays(101, 101, 90.0)
    distance = ray_ground_distance(rays, rotation, (0.0, 0.0, 100.0), 0.0)
    assert distance[50, 0] > distance[50, 50]


def test_upward_rays_return_infinity() -> None:
    """Rays pointing at the sky never hit the ground and return inf."""
    rays = build_rays(5, 5, 90.0)
    distance = ray_ground_distance(rays, np.eye(3), (0.0, 0.0, 100.0), 0.0)
    assert np.all(np.isinf(distance))


def test_level_rays_return_infinity() -> None:
    """Rays parallel to the ground plane return inf instead of dividing by zero."""
    rays = np.zeros((3, 2, 2), dtype=np.float32)
    rays[0] = 1.0
    distance = ray_ground_distance(rays, np.eye(3), (0.0, 0.0, 100.0), 0.0)
    assert np.all(np.isinf(distance))
