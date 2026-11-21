"""V3R worker process: Solv3R streaming inference with JSON-line status on stdout.

The worker reads a JSON config, writes depth maps and point clouds under the run folder, and
emits one JSON object per line. Blender sends the single line "stop" on stdin to end a run.
Every failure is a verbose error; there is no silent fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import torch
from solv3r import Solv3R

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker.config import ConfigError, WorkerConfig
from worker.outputs import create_outputs, write_meta, write_report
from worker.session import SolveSession
from worker.source import FrameSource, StreamLostError


class BackendError(RuntimeError):
    """Raised when the requested compute backend is not usable on this machine."""


class ModelError(RuntimeError):
    """Raised when the Solv3R checkpoint is missing or cannot be loaded."""


class Worker:
    """One worker run: load the model, read frames, publish JSON lines, write outputs."""

    def __init__(self, cfg: WorkerConfig) -> None:
        """Prepare paths and state; the model is loaded in run() so errors become JSON."""
        self.cfg = cfg
        self.outputs = create_outputs(cfg.out_dir)
        self.stop_event = threading.Event()
        self.frames_read = 0
        self.errors: list[str] = []
        self.session: SolveSession | None = None
        self.start_time = time.time()
        self.model_sha256 = ""

    def run(self) -> int:
        """Run the worker and return its exit code."""
        try:
            self._check_device()
            model = self._load_model()
            self.session = SolveSession(self.cfg, model, self.outputs)
            self._emit(self._ready())
            self._watch_stdin()
            return self._run()
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")
            self._emit(self._error(exc))
            self._write_reports()
            return 1

    def _run(self) -> int:
        """Read frames until the source ends, the stream is lost or stop is requested."""
        assert self.session is not None
        source = FrameSource(self.cfg)
        stopped = False
        try:
            for frame in source:
                if self.stop_event.is_set():
                    stopped = True
                    break
                results = self.session.push(frame)
                self.frames_read += 1
                for result in results:
                    self._emit({"type": "frame", **result})
                if self.frames_read % 5 == 0:
                    self._emit(self._status())
            self.session.finish()
            self._write_reports()
            self._emit(self._done(stopped))
            return 0
        except StreamLostError as exc:
            self.errors.append(f"StreamLostError: {exc}")
            self._emit(self._error(exc))
            self.session.finish()
            self._write_reports()
            return 1
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")
            self._emit(self._error(exc))
            self.session.finish()
            self._write_reports()
            return 1
        finally:
            source.close()

    def _check_device(self) -> None:
        """Raise BackendError when the requested device is not actually usable."""
        if self.cfg.device != "cuda":
            return
        if not torch.cuda.is_available():
            raise BackendError(
                "device=cuda requested but torch.cuda.is_available() is False. "
                "Use device=cpu, or install a PyTorch build that supports this GPU."
            )
        properties = torch.cuda.get_device_properties(0)
        arch = getattr(properties, "gcnArchName", "")
        supported = torch.cuda.get_arch_list()
        if arch and arch not in supported:
            raise BackendError(
                f"the installed torch ROCm build does not support {arch}; supported: {supported}. "
                "Use device=cpu or a gfx-compatible torch build."
            )

    def _load_model(self) -> object:
        """Load Solv3R from the local checkpoint, raising ModelError with the expected path."""
        path = Path(self.cfg.checkpoint)
        if not path.is_file():
            raise ModelError(
                f"checkpoint not found: {path}. Download it with: "
                f"curl -L -o {path} https://huggingface.co/jgkwak/solv3r/resolve/main/solv3r.safetensors"
            )
        self.model_sha256 = _sha256(path)
        return Solv3R.from_pretrained(str(path), device=self.cfg.device)

    def _watch_stdin(self) -> None:
        """Set the stop event when Blender sends the line 'stop'."""

        def watch() -> None:
            for line in sys.stdin:
                if line.strip().lower() == "stop":
                    self.stop_event.set()
                    return

        threading.Thread(target=watch, daemon=True).start()

    def _emit(self, message: dict) -> None:
        """Write one compact JSON line to stdout."""
        print(json.dumps(message, separators=(",", ":")), flush=True)

    def _device_name(self) -> str:
        """Return the human-readable device name for reports."""
        if self.cfg.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "CPU"

    def _ready(self) -> dict:
        """Return the ready message sent once the model is loaded."""
        return {
            "type": "ready",
            "device": self.cfg.device,
            "device_name": self._device_name(),
            "model": str(self.cfg.checkpoint),
            "sha256": self.model_sha256,
            "preset": self.cfg.preset,
            "image_long_side": self.cfg.image_long_side,
            "sample_fps": self.cfg.sample_fps,
            "anchors": self.cfg.anchors,
            "recent": self.cfg.recent,
            "memory": self.cfg.memory,
            "calls": self.cfg.calls,
            "depth_enabled": self.cfg.depth_enabled,
            "outputs": str(self.cfg.out_dir),
        }

    def _status(self) -> dict:
        """Return the periodic status message."""
        stats = self.session.stats() if self.session else {}
        return {
            "type": "status",
            "state": "running",
            "frames_read": self.frames_read,
            "frames_answered": stats.get("frames_answered", 0),
            "points": stats.get("points", 0),
            "ms_avg": round(stats.get("ms_avg", 0.0), 1),
            "ms_p95": round(stats.get("ms_p95", 0.0), 1),
            "sample_fps": self.cfg.sample_fps,
        }

    def _done(self, stopped: bool) -> dict:
        """Return the done message sent before exit."""
        stats = self.session.stats() if self.session else {}
        return {
            "type": "done",
            "stopped": stopped,
            "frames_read": self.frames_read,
            "frames_answered": stats.get("frames_answered", 0),
            "points": stats.get("points", 0),
            "ms_avg": round(stats.get("ms_avg", 0.0), 1),
            "elapsed_s": round(time.time() - self.start_time, 1),
            "outputs": str(self.cfg.out_dir),
            "report": str(self.outputs.report_path),
            "cumulative_file": str(self.outputs.cumulative_path),
            "ply_file": str(self.outputs.ply_path),
            "errors": self.errors,
        }

    def _error(self, exc: Exception) -> dict:
        """Return the error message with a traceback."""
        return {
            "type": "error",
            "message": str(exc),
            "type_name": type(exc).__name__,
            "traceback": traceback.format_exc(),
            "fatal": True,
        }

    def _write_reports(self) -> None:
        """Write meta.json and report.md from the collected statistics."""
        stats = self.session.stats() if self.session else {}
        stats.update(
            {
                "frames_read": self.frames_read,
                "elapsed_s": time.time() - self.start_time,
                "device_name": self._device_name(),
            }
        )
        write_meta(self.outputs.meta_path, self.cfg, self.model_sha256, self._device_name(), torch.__version__)
        write_report(self.outputs.report_path, self.cfg, stats, self.errors)


def _sha256(path: Path) -> str:
    """Return the SHA256 of a file in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Parse the config path and run one worker."""
    parser = argparse.ArgumentParser(description="V3R Solv3R worker")
    parser.add_argument("--config", required=True, help="path to the worker JSON config")
    args = parser.parse_args(argv)
    try:
        cfg = WorkerConfig.load(args.config)
    except ConfigError as exc:
        print(json.dumps({"type": "error", "message": str(exc), "fatal": True}), flush=True)
        return 2
    return Worker(cfg).run()


if __name__ == "__main__":
    sys.exit(main())
