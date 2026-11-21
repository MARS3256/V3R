"""Blender-side process manager for the Solv3R worker.

This module spawns the worker with its own Python environment, reads its JSON lines in a
background thread and applies the newest point cloud on a bpy timer. It never imports torch,
cv2 or the worker package, and it never blocks the Blender main thread.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import bpy
import numpy as np

from .point_cloud import apply_points

MAX_MESSAGES_PER_TICK = 60
TIMER_INTERVAL_S = 0.2
STOP_TIMEOUT_S = 10.0


@dataclass
class _Session:
    """One active worker process and the point cloud accumulated from its frames."""

    settings: Any
    process: subprocess.Popen
    messages: queue.Queue = field(default_factory=queue.Queue)
    stderr_lines: deque = field(default_factory=lambda: deque(maxlen=200))
    xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    rgb: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.uint8))
    frames_read: int = 0
    finished: bool = False


_SESSION: _Session | None = None


def is_running() -> bool:
    """Return True while a worker process is alive."""
    return _SESSION is not None and _SESSION.process.poll() is None


def resolve_output_dir(settings: Any) -> Path:
    """Return the absolute run folder selected in the settings."""
    return _worker_paths(settings)["out_dir"]


def start(settings: Any) -> None:
    """Validate the settings, spawn the worker and register the UI timer."""
    global _SESSION
    if is_running():
        raise RuntimeError("a V3R run is already active; stop it first")
    paths = _worker_paths(settings)
    _validate(settings, paths)
    out_dir = Path(paths["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    config = _build_config(settings, paths)
    config_path = out_dir / "worker_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(_repo_root()) + os.pathsep + env.get("PYTHONPATH", "")
    if config["device"] == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    process = subprocess.Popen(
        [str(paths["worker_python"]), str(paths["worker_script"]), "--config", str(config_path)],
        cwd=str(_repo_root()),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )
    session = _Session(settings=settings, process=process)
    _SESSION = session
    threading.Thread(target=_read_stdout, args=(session,), daemon=True).start()
    threading.Thread(target=_read_stderr, args=(session,), daemon=True).start()
    settings.status_text = "Starting worker..."
    settings.error_text = ""
    settings.is_running = True
    bpy.app.timers.register(_poll, first_interval=TIMER_INTERVAL_S)


def stop() -> None:
    """Ask the worker to stop, then wait, terminate or kill it within the timeout."""
    global _SESSION
    session = _SESSION
    if session is None:
        return
    try:
        if session.process.stdin and not session.process.stdin.closed:
            session.process.stdin.write("stop\n")
            session.process.stdin.flush()
    except (BrokenPipeError, OSError):
        pass
    try:
        session.process.wait(timeout=STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        session.process.terminate()
        try:
            session.process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            session.process.kill()
            session.process.wait()
    finally:
        for stream in (session.process.stdin, session.process.stdout, session.process.stderr):
            try:
                if stream and not stream.closed:
                    stream.close()
            except (BrokenPipeError, OSError, ValueError):
                pass
        session.finished = True
        _SESSION = None


def _poll() -> float | None:
    """Drain worker messages, apply the newest frame and keep polling while active."""
    session = _SESSION
    if session is None:
        return None
    latest_frame = None
    for _ in range(MAX_MESSAGES_PER_TICK):
        try:
            message = session.messages.get_nowait()
        except queue.Empty:
            break
        if message.get("type") == "frame":
            latest_frame = message
        else:
            _handle(session, message)
    if latest_frame is not None and not session.finished:
        _apply_frame(session, latest_frame)
    if session.finished:
        return None
    if session.process.poll() is not None:
        _fail(session, f"worker exited with code {session.process.returncode}: {_last_stderr(session)}")
    return TIMER_INTERVAL_S


def _handle(session: _Session, message: dict) -> None:
    """Apply one non-frame message to the scene settings."""
    settings = session.settings
    kind = message.get("type")
    if kind == "ready":
        settings.status_text = (
            f"Ready: {message.get('device_name', '?')} · {message.get('preset', '?')} · "
            f"{message.get('image_long_side', '?')} px"
        )
        settings.device_label = str(message.get("device_name", "?"))
        settings.model_label = Path(message.get("model", "")).name
        settings.is_running = True
    elif kind == "status":
        settings.frames_read = int(message.get("frames_read", 0))
        settings.points_total = int(message.get("points", 0))
        settings.ms_avg = float(message.get("ms_avg", 0.0))
        settings.status_text = (
            message.get("text")
            or f"{settings.frames_read} frames · {settings.points_total} points · {settings.ms_avg:.0f} ms/frame"
        )
    elif kind == "done":
        settings.frames_read = int(message.get("frames_read", 0))
        settings.points_total = int(message.get("points", 0))
        settings.ms_avg = float(message.get("ms_avg", 0.0))
        settings.status_text = f"Done: {settings.points_total} points in {message.get('elapsed_s', 0.0):.1f}s"
        settings.error_text = "; ".join(message.get("errors", []))
        settings.is_running = False
        session.finished = True
        _load_final_cloud(session, message.get("cumulative_file"))
    elif kind == "error":
        _fail(session, str(message.get("message", "worker error")))
        if message.get("traceback"):
            print(message["traceback"])
    elif kind == "__eof__":
        if not session.finished:
            _fail(session, f"worker closed stdout unexpectedly: {_last_stderr(session)}")


def _apply_frame(session: _Session, message: dict) -> None:
    """Load one frame's points, append them to the cloud and update the Blender object."""
    path = message.get("points_file")
    if not path:
        return
    try:
        with np.load(path) as data:
            xyz = data["xyz"].astype(np.float32)
            rgb = data["rgb"].astype(np.uint8)
    except (OSError, KeyError, ValueError) as exc:
        _fail(session, f"cannot read points {path}: {exc}")
        return
    session.xyz = np.concatenate([session.xyz, xyz]) if len(session.xyz) else xyz
    session.rgb = np.concatenate([session.rgb, rgb]) if len(session.rgb) else rgb
    _trim(session)
    apply_points(session.xyz, session.rgb, float(session.settings.point_radius))
    settings = session.settings
    session.frames_read += 1
    settings.frames_read = session.frames_read
    settings.points_total = int(len(session.xyz))
    settings.status_text = (
        f"Frame {message.get('index', '?')}: +{message.get('count', 0)} points · "
        f"total {settings.points_total} · {message.get('ms', 0.0):.0f} ms"
    )
    settings.depth_label = str(message.get("depth_file", ""))


def _load_final_cloud(session: _Session, path: str | None) -> None:
    """Replace the Blender cloud with the worker's final cumulative NPZ."""
    if not path:
        return
    try:
        with np.load(path) as data:
            xyz = data["xyz"].astype(np.float32)
            rgb = data["rgb"].astype(np.uint8)
    except (OSError, KeyError, ValueError) as exc:
        session.settings.error_text = f"cannot read final cloud {path}: {exc}"
        return
    session.xyz = xyz
    session.rgb = rgb
    apply_points(xyz, rgb, float(session.settings.point_radius))


def _trim(session: _Session) -> None:
    """Keep the newest point budget by an evenly spaced subset."""
    budget = int(session.settings.point_budget)
    if len(session.xyz) <= budget:
        return
    idx = np.linspace(0, len(session.xyz) - 1, budget).astype(np.int64)
    session.xyz = session.xyz[idx]
    session.rgb = session.rgb[idx]


def _fail(session: _Session, message: str) -> None:
    """Mark the session failed and show the error in the panel."""
    session.settings.error_text = message
    session.settings.status_text = "Error"
    session.settings.is_running = False
    session.finished = True


def _last_stderr(session: _Session) -> str:
    """Return the last worker stderr lines for an error message."""
    return " | ".join(list(session.stderr_lines)[-4:])


def _read_stdout(session: _Session) -> None:
    """Read JSON lines from the worker and put them on the queue."""
    assert session.process.stdout is not None
    try:
        for line in session.process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                session.messages.put(json.loads(line))
            except json.JSONDecodeError:
                session.stderr_lines.append(f"non-JSON stdout: {line}")
    except (BrokenPipeError, OSError, ValueError):
        pass
    session.messages.put({"type": "__eof__"})


def _read_stderr(session: _Session) -> None:
    """Keep the worker's stderr for error reporting."""
    assert session.process.stderr is not None
    try:
        for line in session.process.stderr:
            session.stderr_lines.append(line.rstrip())
    except (BrokenPipeError, OSError, ValueError):
        pass


def _repo_root() -> Path:
    """Return the repository root that contains worker/ and models/."""
    return Path(__file__).resolve().parents[2]


def _worker_paths(settings: Any) -> dict:
    """Resolve the worker interpreter, script, checkpoint and output directory."""
    root = _repo_root()
    venv_python = root / ".venv-worker" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def resolve(value: str, default: Path) -> Path:
        path = Path(value) if value else default
        return path if path.is_absolute() else root / path

    return {
        "worker_python": resolve(settings.worker_python_path, venv_python),
        "worker_script": resolve(settings.worker_script_path, root / "worker/app.py"),
        "checkpoint": resolve(settings.checkpoint_path, root / "models/solv3r.safetensors"),
        "out_dir": resolve(settings.output_dir, root / "samples/outputs"),
    }


def _validate(settings: Any, paths: dict) -> None:
    """Raise a verbose error when an input path or the worker environment is missing."""
    for name in ("worker_python", "worker_script", "checkpoint"):
        if not Path(paths[name]).is_file():
            raise RuntimeError(f"{name} not found: {paths[name]}")
    source = settings.video_path if settings.source_type == "video" else settings.stream_link
    if not source:
        raise RuntimeError("select a video file or enter a stream link first")
    if settings.source_type == "video" and not Path(source).is_file():
        raise RuntimeError(f"video not found: {source}")
    if settings.source_type == "stream" and not source.lower().startswith(
        ("rtsp://", "rtmp://", "udp://", "srt://", "http://", "https://")
    ):
        raise RuntimeError(
            f"stream link must start with rtsp://, rtmp://, udp://, srt://, http:// or https://: {source}"
        )


def _build_config(settings: Any, paths: dict) -> dict:
    """Build the worker JSON config from the scene settings."""
    source = settings.video_path if settings.source_type == "video" else settings.stream_link
    return {
        "source_kind": settings.source_type,
        "source": source,
        "out_dir": str(paths["out_dir"]),
        "checkpoint": str(paths["checkpoint"]),
        "device": settings.device,
        "preset": settings.preset,
        "image_long_side": int(settings.image_long_side),
        "sample_fps": float(settings.sample_fps),
        "max_frames": int(settings.max_frames),
        "point_stride": int(settings.point_stride),
        "point_budget": int(settings.point_budget),
        "depth_enabled": bool(settings.depth_enabled),
        "stream_lost_timeout_s": float(settings.stream_lost_timeout_s),
        "anchors": int(settings.anchors),
        "recent": int(settings.recent),
        "memory": int(settings.memory),
        "memory_stride": int(settings.memory_stride),
        "calls": int(settings.calls),
        "anchor_calls": int(settings.anchor_calls),
        "refresh_tau": float(settings.refresh_tau),
        "refresh_check": int(settings.refresh_check),
    }
