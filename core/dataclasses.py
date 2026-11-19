"""Frozen dataclasses for data crossing module boundaries, plus worker-to-Blender messages.

This module does NOT encode policy, touch files, or import bpy. Every message is one compact
JSON line; the Blender side sends only the bare line "stop" back to the worker.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray


class MessageError(ValueError):
    """Raised when a message line is invalid JSON or does not match the declared message type."""


@dataclass(frozen=True)
class Message:
    """Base class for one worker-to-Blender JSON message."""

    message_type: ClassVar[str] = ""
    wire_names: ClassVar[dict[str, str]] = {}

    def to_line(self) -> str:
        """Return this message as one compact JSON line including its type tag."""
        data: dict[str, object] = {"type": type(self).message_type}
        for info in fields(self):
            wire = type(self).wire_names.get(info.name, info.name)
            data[wire] = getattr(self, info.name)
        return json.dumps(data, separators=(",", ":"))

    @classmethod
    def from_line(cls, text: str) -> Self:
        """Parse one JSON line into this message class, raising MessageError on any mismatch."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MessageError(f"invalid JSON line: {text!r}") from exc
        if not isinstance(data, dict):
            raise MessageError(f"message line must be a JSON object: {text!r}")
        if data.get("type") != cls.message_type:
            raise MessageError(f"expected type {cls.message_type!r}, got {data.get('type')!r}")
        names = {info.name for info in fields(cls)}
        reverse = {wire: name for name, wire in cls.wire_names.items()}
        kwargs: dict[str, object] = {}
        for wire, value in data.items():
            if wire == "type":
                continue
            name = reverse.get(wire, wire)
            if name not in names:
                raise MessageError(f"unknown field {wire!r} for {cls.message_type}")
            kwargs[name] = value
        try:
            return cls(**kwargs)
        except TypeError as exc:
            raise MessageError(f"bad fields for {cls.message_type}: {exc}") from exc


@dataclass(frozen=True)
class StatusMessage(Message):
    """Periodic run status shown in the Blender panel."""

    message_type: ClassVar[str] = "status"
    state: str
    frames_read: int
    keyframes_used: int
    keyframes_rejected: int
    objects: int
    coverage_pct: float
    text: str


@dataclass(frozen=True)
class OriginMessage(Message):
    """Local East-North-Up origin, published once when the run anchors."""

    message_type: ClassVar[str] = "origin"
    lat: float
    lon: float
    alt_m: float


@dataclass(frozen=True)
class TerrainMessage(Message):
    """A new terrain mesh version and the npz file that holds it."""

    message_type: ClassVar[str] = "terrain"
    version: int
    file: str


@dataclass(frozen=True)
class ObjectUpsertMessage(Message):
    """A new or updated object mesh; the wire field "class" maps to class_name."""

    message_type: ClassVar[str] = "object_upsert"
    wire_names: ClassVar[dict[str, str]] = {"class_name": "class"}
    object_id: int
    version: int
    file: str
    height_m: float
    footprint_area_m2: float
    class_name: str


@dataclass(frozen=True)
class ObjectRemoveMessage(Message):
    """Removal of one tracked object by stable id."""

    message_type: ClassVar[str] = "object_remove"
    object_id: int


@dataclass(frozen=True)
class ErrorMessage(Message):
    """A fatal or non-fatal worker error, never swallowed."""

    message_type: ClassVar[str] = "error"
    code: str
    message: str
    fatal: bool


@dataclass(frozen=True)
class FlightSample:
    """One telemetry sample: video time, WGS84 position and body orientation."""

    t_s: float
    lat_deg: float
    lon_deg: float
    alt_m: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float


@dataclass(frozen=True)
class Frame:
    """One decoded video frame paired with its interpolated telemetry sample."""

    index: int
    t_s: float
    image_bgr: NDArray[np.uint8]
    flight: FlightSample


@dataclass(frozen=True)
class KeyframeResult:
    """The keyframe decision for one frame, including why it was skipped."""

    frame: Frame
    is_keyframe: bool
    reason: str | None
    d_since_last_m: float
    footprint_w_m: float


@dataclass(frozen=True)
class DepthResult:
    """Relative depth map in model units plus the measured inference latency."""

    relative_depth: NDArray[np.float32]
    depth_ms: float


@dataclass(frozen=True)
class ScaleFit:
    """Metric scale fit result: metres per relative-depth unit and its quality."""

    scale: float
    mad_m: float
    ground_frac: float
    ok: bool
    reason: str | None


@dataclass(frozen=True)
class Region:
    """One tracked raised region with its measured height and footprint."""

    object_id: int
    cells: int
    height_m: float
    centroid_e: float
    centroid_n: float
    footprint_area_m2: float
    first_t: float
    last_t: float
    updates: int
