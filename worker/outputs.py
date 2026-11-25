"""Run-folder layout and report writer for Solv3R runs under samples/outputs.

This module does NOT run inference. It creates the output tree and writes meta.json and
report.md from the statistics the worker collects.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from worker.config import WorkerConfig


@dataclass(frozen=True)
class OutputPaths:
    """Every path the worker writes, derived from the run root."""

    root: Path
    depth_dir: Path
    depth_png_dir: Path
    points_dir: Path
    log_path: Path
    meta_path: Path
    report_path: Path
    ply_path: Path
    cumulative_path: Path
    latest_path: Path


def create_outputs(root: str | Path) -> OutputPaths:
    """Create the output tree under root and return its paths."""
    root_path = Path(root).expanduser().resolve()
    paths = OutputPaths(
        root=root_path,
        depth_dir=root_path / "depth",
        depth_png_dir=root_path / "depth_png",
        points_dir=root_path / "points",
        log_path=root_path / "worker.log",
        meta_path=root_path / "meta.json",
        report_path=root_path / "report.md",
        ply_path=root_path / "solv3r_points.ply",
        cumulative_path=root_path / "cumulative.npz",
        latest_path=root_path / "latest.json",
    )
    for directory in (paths.depth_dir, paths.depth_png_dir, paths.points_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def write_json(path: Path, data: dict) -> Path:
    """Write a JSON document with sorted keys and a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_meta(path: Path, cfg: WorkerConfig, model_sha256: str, device_name: str, torch_version: str) -> Path:
    """Write meta.json: config, model hash, device and machine information."""
    meta = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": cfg.__dict__,
        "model_sha256": model_sha256,
        "device_name": device_name,
        "torch_version": torch_version,
        "machine": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "machine": platform.machine(),
        },
    }
    return write_json(path, meta)


def write_report(path: Path, cfg: WorkerConfig, stats: dict, errors: list[str]) -> Path:
    """Write report.md with the run summary, output map and any errors."""
    ms = stats.get("ms_avg", 0.0)
    root = Path(cfg.out_dir)
    lines = [
        "# V3R Solv3R run report",
        "",
        f"- Source: `{cfg.source}` ({cfg.source_kind})",
        f"- Device: `{cfg.device}` ({stats.get('device_name', 'unknown')})",
        f"- Preset: `{cfg.preset}` at {cfg.image_long_side} px long side",
        f"- Frames read: {stats.get('frames_read', 0)}",
        f"- Frames answered: {stats.get('frames_answered', 0)}",
        f"- Points written: {stats.get('points', 0)}",
        f"- Average solver time: {ms:.1f} ms/frame",
        f"- Elapsed: {stats.get('elapsed_s', 0.0):.1f} s",
        "",
        "## Outputs",
        "",
        f"- Depth NPZ: `{root / 'depth'}`",
        f"- Depth PNG: `{root / 'depth_png'}`",
        f"- Per-frame points: `{root / 'points'}`",
        f"- Cumulative cloud: `{root / 'cumulative.npz'}`",
        f"- Cumulative PLY: `{root / 'solv3r_points.ply'}`",
        f"- Latest frame: `{root / 'latest.json'}`",
        "",
    ]
    if errors:
        lines += ["## Errors", ""] + [f"- {error}" for error in errors] + [""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
