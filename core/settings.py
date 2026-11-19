"""Settings registry: every tunable V3R value with a default, allowed range and reason.

This module does NOT read user preferences, files, or Blender state; it is a pure in-memory
registry built fresh on each call so there is no mutable global state (rule R6).
"""

from __future__ import annotations

from dataclasses import dataclass


class SettingsError(ValueError):
    """Raised when a settings key is unknown or a value is outside its allowed range."""


@dataclass(frozen=True)
class Setting:
    """One settings entry: key, label, type, default, unit, allowed range and provenance."""

    key: str
    label: str
    setting_type: type
    default: object
    unit: str | None
    allowed: tuple | None
    provisional: bool
    reason: str


_SPEC: tuple[tuple, ...] = (
    ("fov_x_deg", "Camera FOV", float, 90.0, "deg", (1.0, 179.0), False, "spec assumption; T12 calibrates"),
    ("mount_tilt_deg", "Mount tilt", float, 15.0, "deg", (-45.0, 45.0), False, "team statement; flight check"),
    ("roll_offset_deg", "Roll offset", float, 0.0, "deg", (-180.0, 180.0), False, "team statement; flight check"),
    ("yaw_offset_deg", "Yaw offset", float, 0.0, "deg", (-180.0, 180.0), False, "team statement; flight check"),
    ("image_long_side", "Image long side", int, 1280, "px", (256, 3840), False, "VRAM headroom; Gate A may raise"),
    ("overlap_ratio", "Overlap ratio", float, 0.65, "ratio", (0.0, 0.95), False, "standard mapping practice"),
    ("max_time_gap_s", "Max time gap", float, 6.0, "s", (0.1, 60.0), False, "safety net if GPS speed missing"),
    ("blur_margin_ratio", "Blur margin", float, 0.5, "ratio", (0.0, 1.0), False, "conservative; T12 measures"),
    ("max_tilt_deg", "Max tilt", float, 45.0, "deg", (0.0, 90.0), False, "conservative; T12 measures"),
    (
        "min_ground_fraction",
        "Min ground fraction",
        float,
        0.20,
        "ratio",
        (0.0, 1.0),
        False,
        "below this the fit is untrustworthy",
    ),
    ("max_fit_mad_m", "Max fit MAD", float, 3.0, "m", (0.0, None), False, "generous vs GPS vertical noise"),
    ("smoothness_window", "Smoothness window", int, 9, "px", (3, 51), True, "provisional; T12 measures"),
    ("smoothness_max_var", "Max smoothness var", float, 0.01, "var", (0.0, 1.0), True, "provisional; T12 measures"),
    (
        "outlier_mad_multiplier",
        "Outlier MAD factor",
        float,
        3.0,
        "ratio",
        (0.0, None),
        True,
        "provisional; T12 measures",
    ),
    (
        "ground_region_row_fraction",
        "Ground row fraction",
        float,
        0.5,
        "ratio",
        (0.0, 1.0),
        True,
        "provisional; T12 measures",
    ),
    (
        "ground_ref_settle_keyframes",
        "Ground settle frames",
        int,
        10,
        "frames",
        (1, 1000),
        False,
        "before feedback loop engages",
    ),
    ("ground_filter_kernel", "Ground filter kernel", int, 5, "px", (1, 99), True, "provisional; T12 measures"),
    ("cell_factor", "Cell factor", int, 4, "factor", (1, 64), False, "plan 4.4.6; bounds memory"),
    ("min_hits", "Min hits", int, 3, "hits", (1, 1000), False, "unknown cells stay unknown"),
    ("area_limit_km2", "Area limit", float, 4.0, "km2", (0.01, None), False, "plan 4.4.6; bounds memory"),
    ("min_object_height_m", "Min object height", float, 2.0, "m", (0.0, None), False, "visible-structure floor"),
    ("min_cells", "Min cells", int, 25, "cells", (1, 1000), False, "visible-structure floor"),
    ("keep_overlap", "Keep overlap", float, 0.3, "ratio", (0.0, 1.0), False, "stable ID tracking"),
    ("remove_after_updates", "Remove after updates", int, 5, "updates", (1, 1000), False, "stable ID tracking"),
    (
        "min_region_confidence",
        "Min region confidence",
        float,
        0.5,
        "ratio",
        (0.0, 1.0),
        True,
        "provisional; T12 measures",
    ),
    ("approx_poly_epsilon", "Simplify epsilon", float, 1.5, "m", (0.0, None), True, "provisional; T12 measures"),
    ("max_updates_per_tick", "Max updates per tick", int, 6, "updates", (1, 1000), False, "NFR-1 budget"),
    ("tick_budget_ms", "Tick budget", int, 30, "ms", (1, 1000), False, "NFR-1 budget"),
    ("stop_timeout_s", "Stop timeout", float, 10.0, "s", (0.1, None), False, "NFR-1; Stop ends the worker"),
    ("probe_timeout_s", "Probe timeout", float, 5.0, "s", (0.1, None), False, "plan 4.6 stream probe"),
    ("lost_timeout_s", "Lost timeout", float, 10.0, "s", (0.1, None), False, "plan 4.6 stream probe"),
    ("queue_len", "Queue length", int, 8, "frames", (1, 1024), False, "plan 4.6 drop-oldest ingest"),
    ("diff_pixel_threshold", "Diff threshold", int, 8, "level", (0, 255), True, "provisional; 0-255 scale"),
    (
        "gps_consistency_tolerance_m",
        "GPS consistency",
        float,
        5.0,
        "m",
        (0.0, None),
        True,
        "provisional; consumer-GPS bound",
    ),
    (
        "compute_backend",
        "Compute backend",
        str,
        "directml",
        None,
        ("directml", "ncnn_vulkan", "cpu_reference"),
        False,
        "Gate A decides",
    ),
    ("worker_python_path", "Worker Python", str, "python", None, None, False, "override with venv interpreter"),
    (
        "debug_artifacts",
        "Debug artifacts",
        str,
        "always_first_run",
        None,
        ("on", "off", "always_first_run"),
        False,
        "plan 5.5 artifact cost",
    ),
)


def registry() -> tuple[Setting, ...]:
    """Return every settings entry in declaration order."""
    return tuple(Setting(*spec) for spec in _SPEC)


def defaults() -> dict[str, object]:
    """Return a fresh dict of every default value, keyed by setting key."""
    return {entry.key: entry.default for entry in registry()}


def describe() -> list[Setting]:
    """Return every settings entry in declaration order."""
    return list(registry())


def validate(overrides: dict[str, object] | None = None) -> dict[str, object]:
    """Return defaults merged with overrides, raising SettingsError on unknown or bad values."""
    entries = {entry.key: entry for entry in registry()}
    values: dict[str, object] = {key: entry.default for key, entry in entries.items()}
    for key, value in (overrides or {}).items():
        entry = entries.get(key)
        if entry is None:
            raise SettingsError(f"unknown setting key: {key}")
        values[key] = _checked_value(entry, value)
    return values


def _checked_value(entry: Setting, value: object) -> object:
    """Return the value coerced to the entry type, or raise SettingsError."""
    kind = entry.setting_type
    if kind is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SettingsError(f"{entry.key} must be a number, got {value!r}")
        value = float(value)
    elif kind is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SettingsError(f"{entry.key} must be an integer, got {value!r}")
    elif not isinstance(value, str):
        raise SettingsError(f"{entry.key} must be a string, got {value!r}")
    _check_allowed(entry, value)
    return value


def _check_allowed(entry: Setting, value: object) -> None:
    """Raise SettingsError when the value is outside the entry's allowed range."""
    allowed = entry.allowed
    if allowed is None:
        return
    if entry.setting_type is str:
        if value not in allowed:
            raise SettingsError(f"{entry.key} must be one of {allowed}, got {value!r}")
        return
    low, high = allowed
    if low is not None and value < low:
        raise SettingsError(f"{entry.key} must be >= {low}, got {value!r}")
    if high is not None and value > high:
        raise SettingsError(f"{entry.key} must be <= {high}, got {value!r}")
