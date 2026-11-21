"""Depth-map to point-cloud conversion and depth-map image/NPZ writers.

This module does NOT read frames or run the model. It is pure numpy plus OpenCV colour mapping,
so the same code is used by the CLI and the Blender worker.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def depth_edges(depth: np.ndarray, threshold: float = 0.04) -> np.ndarray:
    """Return a boolean mask of depth discontinuities (relative central difference above threshold)."""
    gx = np.zeros_like(depth)
    gy = np.zeros_like(depth)
    gx[:, 1:-1] = np.abs(depth[:, 2:] - depth[:, :-2]) / 2.0
    gy[1:-1, :] = np.abs(depth[2:, :] - depth[:-2, :]) / 2.0
    g = np.zeros_like(depth)
    g[:, 1:-1] = gx[:, 1:-1]
    g[1:-1, :] = np.maximum(g[1:-1, :], gy[1:-1, :])
    return g / np.maximum(depth, 1e-6) > threshold


def depth_to_points(
    depth: np.ndarray,
    conf: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
    rgb: np.ndarray,
    stride: int = 3,
    conf_keep: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return world-space points (N, 3) float32 and their colours (N, 3) uint8.

    Points on depth edges and the least confident half are dropped; rgb is (H, W, 3) uint8 at the
    depth resolution. K and c2w are the Solv3R camera at the same resolution.
    """
    h, w = depth.shape
    ys, xs = np.mgrid[stride // 2 : h : stride, stride // 2 : w : stride]
    ys, xs = ys.ravel(), xs.ravel()
    z = depth[ys, xs]
    cf = conf[ys, xs]
    ok = np.isfinite(z) & (z > 0) & ~depth_edges(depth)[ys, xs]
    if ok.any():
        ok &= cf >= np.quantile(cf[ok], 1.0 - conf_keep)
    idx = np.nonzero(ok)[0]
    zz = z[idx]
    x = (xs[idx] + 0.5 - K[0, 2]) / K[0, 0] * zz
    y = (ys[idx] + 0.5 - K[1, 2]) / K[1, 1] * zz
    cam = np.stack([x, y, zz], axis=1)
    world = cam @ c2w[:3, :3].T + c2w[:3, 3]
    colours = rgb[ys[idx], xs[idx]]
    return world.astype(np.float32), colours.astype(np.uint8)


def depth_to_colour(depth: np.ndarray) -> np.ndarray:
    """Return an RGB uint8 colourisation of a depth map (turbo, 2nd-98th percentile stretch)."""
    import cv2

    d = np.asarray(depth, np.float32)
    finite = np.isfinite(d) & (d > 0)
    out = np.zeros((*d.shape, 3), np.uint8)
    if not finite.any():
        return out
    lo, hi = np.percentile(d[finite], [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    coloured = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[finite] = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)[finite]
    return out


def save_depth(path: Path, depth: np.ndarray, conf: np.ndarray, c2w: np.ndarray, K: np.ndarray) -> Path:
    """Write depth, confidence, camera-to-world and intrinsics as a compressed NPZ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        depth=np.asarray(depth, np.float16),
        conf=np.asarray(conf, np.float16),
        c2w=np.asarray(c2w, np.float32),
        K=np.asarray(K, np.float32),
    )
    return path


def save_points(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> Path:
    """Write world-space points and colours as a compressed NPZ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, xyz=np.asarray(xyz, np.float32), rgb=np.asarray(rgb, np.uint8))
    return path


def load_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load points written by save_points."""
    with np.load(path) as data:
        return data["xyz"].astype(np.float32), data["rgb"].astype(np.uint8)


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> Path:
    """Write a binary little-endian PLY with per-point colour."""
    from solv3r.export import write_ply as _write_ply

    _write_ply(str(path), np.asarray(xyz, np.float32), np.asarray(rgb, np.uint8))
    return path
