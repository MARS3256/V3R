"""Tests for depth-to-point conversion and point-cloud file round-trips."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from worker.points import depth_edges, depth_to_colour, depth_to_points, load_points, save_points

K = np.array([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]], np.float32)
C2W = np.eye(4, dtype=np.float32)


def test_depth_to_points_flat_plane() -> None:
    """A constant depth plane produces a regular grid at the expected metric coordinates."""
    depth = np.full((10, 10), 2.0, np.float32)
    conf = np.ones((10, 10), np.float32)
    rgb = np.zeros((10, 10, 3), np.uint8)
    xyz, colors = depth_to_points(depth, conf, C2W, K, rgb, stride=2, conf_keep=1.0)
    assert xyz.shape == (25, 3)
    assert colors.shape == (25, 3)
    assert np.allclose(xyz[:, 2], 2.0)
    expected_x = np.tile((np.array([1, 3, 5, 7, 9]) + 0.5 - 5.0) / 10.0 * 2.0, 5)
    assert np.allclose(xyz[:, 0], expected_x)


def test_conf_keep_drops_low_confidence() -> None:
    """conf_keep=0.5 keeps at most half of the candidate points."""
    depth = np.full((10, 10), 2.0, np.float32)
    conf = np.linspace(0.0, 1.0, 100).reshape(10, 10).astype(np.float32)
    rgb = np.zeros((10, 10, 3), np.uint8)
    xyz, _ = depth_to_points(depth, conf, C2W, K, rgb, stride=1, conf_keep=0.5)
    assert 0 < len(xyz) <= 50


def test_depth_edges_detects_step() -> None:
    """A depth step is flagged as an edge."""
    depth = np.ones((10, 10), np.float32)
    depth[:, 5:] = 5.0
    edges = depth_edges(depth)
    assert edges[:, 4:6].any()


def test_depth_to_colour_shape() -> None:
    """The colourised depth is an RGB uint8 image."""
    depth = np.linspace(1.0, 5.0, 100).reshape(10, 10).astype(np.float32)
    colour = depth_to_colour(depth)
    assert colour.shape == (10, 10, 3)
    assert colour.dtype == np.uint8


def test_points_round_trip(tmp_path: Path) -> None:
    """Points and colours survive save_points and load_points."""
    xyz = np.array([[0, 0, 0], [1, 2, 3]], np.float32)
    rgb = np.array([[255, 0, 0], [0, 255, 0]], np.uint8)
    path = tmp_path / "points.npz"
    save_points(path, xyz, rgb)
    loaded_xyz, loaded_rgb = load_points(path)
    assert np.allclose(loaded_xyz, xyz)
    assert np.array_equal(loaded_rgb, rgb)
