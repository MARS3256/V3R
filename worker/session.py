"""One Solv3R streaming session: frames in, depth maps and points out.

This module does NOT open the source or parse stdin. It owns the StreamingSolver, keeps the
model images needed for colours, accumulates the point cloud and writes the output files.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from solv3r import StreamingSolver, load_images  # noqa: F401  (kept for API parity)
from solv3r.images import model_size, resize
from solv3r.streaming import StreamConfig

from worker.config import WorkerConfig
from worker.outputs import OutputPaths, write_json
from worker.points import depth_to_colour, depth_to_points, save_depth, save_points, write_ply
from worker.source import Frame


class SolveSession:
    """Hold the solver, the per-frame images and the cumulative cloud for one run."""

    def __init__(self, cfg: WorkerConfig, model: object, outputs: OutputPaths) -> None:
        """Create a streaming solver with the configured anchors, memory and call counts."""
        self.cfg = cfg
        self.model = model
        self.outputs = outputs
        stream_cfg = StreamConfig(
            anchors=cfg.anchors,
            recent=cfg.recent,
            memory=cfg.memory,
            memory_stride=cfg.memory_stride,
            calls=cfg.calls,
            anchor_calls=cfg.anchor_calls,
            refresh_tau=cfg.refresh_tau,
            refresh_check=cfg.refresh_check,
        )
        self.solver = StreamingSolver(model, stream_cfg, depth=True)
        self.images: dict[int, torch.Tensor] = {}
        self.cumulative_xyz: list[np.ndarray] = []
        self.cumulative_rgb: list[np.ndarray] = []
        self.total_points = 0
        self.frames_answered = 0
        self.model_image_size: tuple[int, int] | None = None
        self.ms_list: list[float] = []
        self.last_result: dict = {}

    def push(self, frame: Frame) -> list[dict]:
        """Push one frame and return one result dict per frame the solver answered."""
        image = self._model_image(frame)
        self.images[frame.index] = image
        start = time.perf_counter()
        answered = self.solver.push(image)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if answered:
            self.ms_list.append(elapsed_ms)
        return [self._frame_result(f, elapsed_ms) for f in answered]

    def finish(self) -> None:
        """Write the final cumulative NPZ and PLY."""
        self._save_cumulative()
        self._save_ply()

    def stats(self) -> dict:
        """Return latency and point statistics for the report."""
        ordered = sorted(self.ms_list)
        p95 = ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))] if ordered else 0.0
        return {
            "ms_avg": float(np.mean(self.ms_list)) if self.ms_list else 0.0,
            "ms_p95": float(p95),
            "frames_answered": self.frames_answered,
            "points": self.total_points,
        }

    def _model_image(self, frame: Frame) -> torch.Tensor:
        """Convert one RGB frame to the model input tensor at the configured long side."""
        if self.model_image_size is None:
            self.model_image_size = model_size(frame.rgb.shape[0], frame.rgb.shape[1], self.cfg.image_long_side)
        image = torch.from_numpy(frame.rgb).permute(2, 0, 1).float() / 255.0
        return resize(image[None], self.model_image_size)[0]

    def _frame_result(self, f: int, elapsed_ms: float) -> dict:
        """Save one answered frame and return its message fields."""
        depth = self.solver.depths[f]
        conf = self.solver.confs[f]
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = self.solver.R[f].float().numpy()
        c2w[:3, 3] = self.solver.C[f].float().numpy()
        K = self.solver.K[f].float().numpy()
        image = self.images.get(f)
        if image is None:
            raise RuntimeError(f"no model image kept for answered frame {f}")
        rgb = (image.permute(1, 2, 0).clamp(0, 1) * 255).round().byte().numpy()
        xyz, colours = depth_to_points(depth, conf, c2w, K, rgb, self.cfg.point_stride)
        depth_path = (
            save_depth(self.outputs.depth_dir / f"frame_{f:06d}.npz", depth, conf, c2w, K)
            if self.cfg.depth_enabled
            else None
        )
        png_path = self._write_depth_png(f, depth) if self.cfg.depth_enabled else None
        points_path = save_points(self.outputs.points_dir / f"points_{f:06d}.npz", xyz, colours)
        self._append_cumulative(xyz, colours)
        self.frames_answered += 1
        if self.frames_answered % self.cfg.cumulative_every == 0:
            self._save_cumulative()
        if self.frames_answered % self.cfg.ply_every == 0:
            self._save_ply()
        self.last_result = {
            "index": f,
            "points_file": str(points_path),
            "depth_file": str(depth_path) if depth_path else None,
            "depth_png": str(png_path) if png_path else None,
            "count": int(len(xyz)),
            "total": int(self.total_points),
            "c2w": c2w.tolist(),
            "K": K.tolist(),
            "ms": round(elapsed_ms, 1),
        }
        write_json(self.outputs.latest_path, self.last_result)
        return self.last_result

    def _write_depth_png(self, f: int, depth: np.ndarray) -> Path | None:
        """Write a colourised depth PNG when depth output is enabled."""
        if not self.cfg.depth_enabled:
            return None
        path = self.outputs.depth_png_dir / f"frame_{f:06d}.png"
        image = cv2.cvtColor(depth_to_colour(depth), cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"could not write depth PNG {path}")
        return path

    def _append_cumulative(self, xyz: np.ndarray, rgb: np.ndarray) -> None:
        """Add points to the cumulative cloud and trim it to the configured budget."""
        self.cumulative_xyz.append(xyz)
        self.cumulative_rgb.append(rgb)
        self.total_points += len(xyz)
        if self.total_points > self.cfg.point_budget:
            self._trim()

    def _trim(self) -> None:
        """Keep an evenly spaced subset of the cumulative cloud within the budget."""
        xyz = np.concatenate(self.cumulative_xyz)
        rgb = np.concatenate(self.cumulative_rgb)
        idx = np.linspace(0, len(xyz) - 1, self.cfg.point_budget).astype(np.int64)
        self.cumulative_xyz = [xyz[idx]]
        self.cumulative_rgb = [rgb[idx]]
        self.total_points = int(self.cfg.point_budget)

    def _cumulative_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the cumulative points and colours as two arrays."""
        if not self.cumulative_xyz:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
        return np.concatenate(self.cumulative_xyz), np.concatenate(self.cumulative_rgb)

    def _save_cumulative(self) -> None:
        """Write the cumulative point cloud NPZ."""
        xyz, rgb = self._cumulative_arrays()
        save_points(self.outputs.cumulative_path, xyz, rgb)

    def _save_ply(self) -> None:
        """Write the cumulative point cloud PLY."""
        xyz, rgb = self._cumulative_arrays()
        if len(xyz):
            write_ply(self.outputs.ply_path, xyz, rgb)
