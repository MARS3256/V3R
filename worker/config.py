"""Worker configuration: load and validate the JSON config written by Blender or the CLI.

This module does NOT read video, load models or touch the GPU. It only turns a JSON object
into a validated WorkerConfig, raising ConfigError with the offending field.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

VALID_DEVICES = ("cpu", "cuda")
VALID_PRESETS = ("S4", "L4", "L8")
VALID_LONG_SIDES = (252, 504)
VALID_SOURCE_KINDS = ("video", "stream")


class ConfigError(ValueError):
    """Raised when the worker config is missing, malformed or out of range."""


@dataclass(frozen=True)
class WorkerConfig:
    """Every setting the worker needs, with the defaults used by the CLI and the plugin."""

    source_kind: str
    source: str
    out_dir: str
    checkpoint: str
    device: str = "cpu"
    preset: str = "S4"
    image_long_side: int = 252
    sample_fps: float = 2.0
    max_frames: int = 0
    point_stride: int = 3
    point_budget: int = 500_000
    depth_enabled: bool = True
    stream_lost_timeout_s: float = 10.0
    anchors: int = 4
    recent: int = 4
    memory: int = 16
    memory_stride: int = 16
    calls: int = 4
    anchor_calls: int = 16
    refresh_tau: float = 0.02
    refresh_check: int = 4
    ply_every: int = 10
    cumulative_every: int = 5
    report: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> WorkerConfig:
        """Build a config from a dict, rejecting unknown keys and invalid values."""
        if not isinstance(data, dict):
            raise ConfigError(f"config must be a JSON object, got {type(data).__name__}")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(f"unknown config keys: {', '.join(unknown)}")
        missing = [name for name in ("source_kind", "source", "out_dir", "checkpoint") if not data.get(name)]
        if missing:
            raise ConfigError(f"missing required config keys: {', '.join(missing)}")
        cfg = cls(**data)
        cfg.validate()
        return cls(
            **{
                **{f.name: getattr(cfg, f.name) for f in fields(cls)},
                "out_dir": str(Path(cfg.out_dir).expanduser().resolve()),
                "checkpoint": str(Path(cfg.checkpoint).expanduser().resolve()),
            }
        )

    @classmethod
    def load(cls, path: str) -> WorkerConfig:
        """Load a config from a JSON file, raising ConfigError with the file path."""
        config_path = Path(path)
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ConfigError(f"cannot read config {config_path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid JSON in {config_path}: {exc}") from exc
        return cls.from_dict(data)

    def validate(self) -> None:
        """Raise ConfigError when any field is outside its allowed range."""
        if self.source_kind not in VALID_SOURCE_KINDS:
            raise ConfigError(f"source_kind must be one of {VALID_SOURCE_KINDS}, got {self.source_kind!r}")
        if self.device not in VALID_DEVICES:
            raise ConfigError(f"device must be one of {VALID_DEVICES}, got {self.device!r}")
        if self.preset not in VALID_PRESETS:
            raise ConfigError(f"preset must be one of {VALID_PRESETS}, got {self.preset!r}")
        if self.image_long_side not in VALID_LONG_SIDES:
            raise ConfigError(f"image_long_side must be one of {VALID_LONG_SIDES}, got {self.image_long_side!r}")
        if self.sample_fps < 0:
            raise ConfigError(f"sample_fps must be >= 0, got {self.sample_fps!r}")
        if self.max_frames < 0:
            raise ConfigError(f"max_frames must be >= 0, got {self.max_frames!r}")
        if self.point_stride < 1:
            raise ConfigError(f"point_stride must be >= 1, got {self.point_stride!r}")
        if self.point_budget < 1000:
            raise ConfigError(f"point_budget must be >= 1000, got {self.point_budget!r}")
        if self.stream_lost_timeout_s <= 0:
            raise ConfigError(f"stream_lost_timeout_s must be > 0, got {self.stream_lost_timeout_s!r}")
        for name, low in (
            ("anchors", 1),
            ("recent", 1),
            ("memory", 0),
            ("memory_stride", 1),
            ("calls", 1),
            ("anchor_calls", 1),
            ("refresh_check", 1),
            ("ply_every", 1),
            ("cumulative_every", 1),
        ):
            if getattr(self, name) < low:
                raise ConfigError(f"{name} must be >= {low}, got {getattr(self, name)!r}")
        if self.refresh_tau < 0:
            raise ConfigError(f"refresh_tau must be >= 0, got {self.refresh_tau!r}")
