"""Contract tests: message round-trips, telemetry column order and settings validation."""

from __future__ import annotations

import pytest

from core import settings
from core.dataclasses import (
    ErrorMessage,
    Message,
    MessageError,
    ObjectRemoveMessage,
    ObjectUpsertMessage,
    OriginMessage,
    StatusMessage,
    TerrainMessage,
)
from core.debugio import TELEMETRY_COLUMNS

_MESSAGES: tuple[Message, ...] = (
    StatusMessage("building", 10, 2, 1, 3, 42.5, "ok"),
    OriginMessage(23.8, 90.4, 50.0),
    TerrainMessage(1, "terrain/step_0001.npz"),
    ObjectUpsertMessage(1, 2, "objects/step_0001/obj_0001.npz", 12.0, 34.5, "unknown_structure"),
    ObjectRemoveMessage(7),
    ErrorMessage("E1", "boom", True),
)

_EXPECTED_COLUMNS = (
    "frame_index",
    "t_video_s",
    "lat",
    "lon",
    "alt",
    "roll",
    "pitch",
    "yaw",
    "h_agl_m",
    "footprint_w_m",
    "d_since_last_m",
    "blur_score",
    "is_keyframe",
    "skip_reason",
    "depth_ms",
    "fit_scale",
    "fit_mad_m",
    "ground_frac",
    "cloud_pts",
    "fuse_ms",
    "regions",
    "mesh_ms",
    "total_ms",
    "gpu_mem_mb",
)


@pytest.mark.parametrize("message", _MESSAGES)
def test_message_round_trip(message: Message) -> None:
    """Every message survives to_line then from_line unchanged."""
    line = message.to_line()
    assert type(message).from_line(line) == message


def test_object_upsert_uses_wire_field_class() -> None:
    """The object_upsert payload carries "class", never the Python name class_name."""
    line = _MESSAGES[3].to_line()
    assert '"class":"unknown_structure"' in line
    assert "class_name" not in line


def test_from_line_rejects_wrong_type() -> None:
    """A status parser refuses an origin line instead of silently accepting it."""
    origin = OriginMessage(1.0, 2.0, 3.0)
    with pytest.raises(MessageError):
        StatusMessage.from_line(origin.to_line())


def test_from_line_rejects_unknown_field() -> None:
    """An extra field is an error, not something to ignore."""
    with pytest.raises(MessageError):
        StatusMessage.from_line('{"type":"status","nope":1}')


def test_telemetry_columns_exact_order() -> None:
    """TELEMETRY_COLUMNS equals the plan 5.3 list, in order."""
    assert TELEMETRY_COLUMNS == _EXPECTED_COLUMNS


def test_defaults_contain_expected_values() -> None:
    """The registry ships the card's defaults."""
    values = settings.defaults()
    assert values["fov_x_deg"] == 90.0
    assert values["image_long_side"] == 1280
    assert values["compute_backend"] == "directml"
    assert values["debug_artifacts"] == "always_first_run"


def test_validate_rejects_unknown_key() -> None:
    """An unknown key raises SettingsError."""
    with pytest.raises(settings.SettingsError):
        settings.validate({"not_a_setting": 1})


def test_validate_rejects_out_of_range_value() -> None:
    """A value above its allowed maximum raises SettingsError."""
    with pytest.raises(settings.SettingsError):
        settings.validate({"fov_x_deg": 500.0})


def test_validate_rejects_wrong_type() -> None:
    """A string where a number is expected raises SettingsError."""
    with pytest.raises(settings.SettingsError):
        settings.validate({"image_long_side": "big"})


def test_validate_merges_defaults_and_coerces_int_to_float() -> None:
    """Overrides merge over defaults and an int is accepted where a float is declared."""
    values = settings.validate({"fov_x_deg": 75, "image_long_side": 2048})
    assert values["fov_x_deg"] == 75.0
    assert values["image_long_side"] == 2048
    assert values["overlap_ratio"] == 0.65


def test_describe_returns_frozen_settings() -> None:
    """describe() returns Setting entries in declaration order."""
    described = settings.describe()
    assert described[0].key == "fov_x_deg"
    assert all(isinstance(entry, settings.Setting) for entry in described)
    assert all(entry.reason for entry in described)
