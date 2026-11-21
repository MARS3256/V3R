"""Tests for worker configuration validation and path resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from worker.config import ConfigError, WorkerConfig

_VALID = {
    "source_kind": "video",
    "source": "clip.mp4",
    "out_dir": "samples/outputs/run",
    "checkpoint": "models/solv3r.safetensors",
}


def test_defaults_are_valid() -> None:
    """A minimal config loads with the documented defaults."""
    cfg = WorkerConfig.from_dict(_VALID)
    assert cfg.device == "cpu"
    assert cfg.preset == "S4"
    assert cfg.image_long_side == 252
    assert cfg.sample_fps == 2.0
    assert cfg.depth_enabled is True
    assert cfg.point_budget == 500_000


def test_unknown_key_raises() -> None:
    """An unknown key is rejected instead of ignored."""
    with pytest.raises(ConfigError):
        WorkerConfig.from_dict({**_VALID, "nope": 1})


def test_missing_key_raises() -> None:
    """A missing required key is rejected."""
    data = dict(_VALID)
    data.pop("source")
    with pytest.raises(ConfigError):
        WorkerConfig.from_dict(data)


def test_out_of_range_raises() -> None:
    """An out-of-range value or an unknown device is rejected."""
    with pytest.raises(ConfigError):
        WorkerConfig.from_dict({**_VALID, "sample_fps": -1})
    with pytest.raises(ConfigError):
        WorkerConfig.from_dict({**_VALID, "image_long_side": 333})
    with pytest.raises(ConfigError):
        WorkerConfig.from_dict({**_VALID, "device": "vulkan"})


def test_load_from_json(tmp_path: Path) -> None:
    """A config file is loaded and its paths are resolved to absolute paths."""
    path = tmp_path / "worker.json"
    path.write_text(json.dumps(_VALID), encoding="utf-8")
    cfg = WorkerConfig.load(str(path))
    assert Path(cfg.out_dir).is_absolute()
    assert Path(cfg.checkpoint).is_absolute()


def test_load_missing_file_raises(tmp_path: Path) -> None:
    """A missing config file raises ConfigError with the path."""
    with pytest.raises(ConfigError):
        WorkerConfig.load(str(tmp_path / "missing.json"))
