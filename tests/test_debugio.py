"""Tests for the run-folder tree, CSV appender, PNG gate and meta.json."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.debugio import (
    TELEMETRY_COLUMNS,
    CsvAppender,
    RunFolder,
    RunFolderExistsError,
    create_run_folder,
    input_fingerprint,
    png_gate,
    write_meta,
)

_EXPECTED_DIRS = (
    "ingest",
    "ingest/keyframes",
    "depth",
    "depth/raw",
    "depth/png",
    "diff",
    "ground",
    "cloud",
    "cloud/text",
    "heightmap",
    "heightmap/png",
    "objects",
    "detect",
    "graphs",
)


def test_tree_contains_every_directory(tmp_path: Path) -> None:
    """create_run_folder makes the full plan 5.1 directory tree."""
    run = create_run_folder(tmp_path, "run-001")
    for relative in _EXPECTED_DIRS:
        assert (run.root / relative).is_dir(), relative


def test_create_refuses_existing_folder(tmp_path: Path) -> None:
    """A second create with the same run id raises RunFolderExistsError."""
    create_run_folder(tmp_path, "run-001")
    with pytest.raises(RunFolderExistsError):
        create_run_folder(tmp_path, "run-001")


def test_path_helpers_use_documented_families(tmp_path: Path) -> None:
    """Frame paths are zero-padded and live in the documented directories."""
    run = create_run_folder(tmp_path, "run-001")
    assert run.keyframe_jpg(123).name == "frame_000123.jpg"
    assert run.depth_raw_npz(7).parent.name == "raw"
    assert run.depth_png(7).parent.name == "png"
    assert run.heightmap_png(7).name == "step_0007.png"
    assert run.objects_dir(7).is_dir()
    assert run.telemetry_csv().name == "telemetry.csv"
    assert run.meta_json().name == "meta.json"


def test_csv_header_written_once(tmp_path: Path) -> None:
    """The header is written once even across close and reopen."""
    path = tmp_path / "telemetry.csv"
    appender = CsvAppender(path, TELEMETRY_COLUMNS)
    appender.append_row([0] * len(TELEMETRY_COLUMNS))
    appender.close()
    appender = CsvAppender(path, TELEMETRY_COLUMNS)
    appender.append_row([1] * len(TELEMETRY_COLUMNS))
    appender.close()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(TELEMETRY_COLUMNS)
    assert sum(line.startswith("frame_index") for line in lines) == 1
    assert len(lines) == 3


def test_csv_append_rejects_wrong_width(tmp_path: Path) -> None:
    """A row with the wrong number of columns is an error, not a silent write."""
    appender = CsvAppender(tmp_path / "x.csv", ("a", "b"))
    with pytest.raises(ValueError):
        appender.append_row([1])
    appender.close()


def _write_meta(run: RunFolder, fingerprint: str) -> None:
    """Write a minimal meta.json carrying the given input fingerprint."""
    write_meta(run, {"fov_x_deg": 90.0}, {"depth": "abc"}, fingerprint, {"v3r": "0.1.0"})


def test_png_gate_first_run(tmp_path: Path) -> None:
    """The first run for a fingerprint writes PNGs."""
    run = create_run_folder(tmp_path, "run-001")
    assert png_gate(run, "always_first_run", "fp-1") is True


def test_png_gate_second_run_same_fingerprint(tmp_path: Path) -> None:
    """A later run with the same fingerprint does not write PNGs."""
    _write_meta(create_run_folder(tmp_path, "run-001"), "fp-1")
    second = create_run_folder(tmp_path, "run-002")
    assert png_gate(second, "always_first_run", "fp-1") is False


def test_png_gate_mode_on_overrides_siblings(tmp_path: Path) -> None:
    """Mode "on" writes PNGs even when a sibling has the same fingerprint."""
    _write_meta(create_run_folder(tmp_path, "run-001"), "fp-1")
    second = create_run_folder(tmp_path, "run-002")
    assert png_gate(second, "on", "fp-1") is True


def test_png_gate_mode_off(tmp_path: Path) -> None:
    """Mode "off" never writes PNGs."""
    run = create_run_folder(tmp_path, "run-001")
    assert png_gate(run, "off", "fp-1") is False


def test_png_gate_skips_unreadable_sibling(tmp_path: Path) -> None:
    """An unreadable sibling meta.json is skipped and does not count as a previous run."""
    first = create_run_folder(tmp_path, "run-001")
    first.meta_json().write_text("not json", encoding="utf-8")
    second = create_run_folder(tmp_path, "run-002")
    assert png_gate(second, "always_first_run", "fp-1") is True


def test_png_gate_rejects_unknown_mode(tmp_path: Path) -> None:
    """An unknown debug_artifacts mode raises instead of guessing."""
    run = create_run_folder(tmp_path, "run-001")
    with pytest.raises(ValueError):
        png_gate(run, "sometimes", "fp-1")


def test_meta_round_trips_settings(tmp_path: Path) -> None:
    """write_meta records settings, hashes, fingerprint and versions."""
    run = create_run_folder(tmp_path, "run-001")
    snapshot = {"fov_x_deg": 90.0, "compute_backend": "directml"}
    path = write_meta(run, snapshot, {"depth": "abc"}, "fp-1", {"v3r": "0.1.0"})
    meta = json.loads(path.read_text(encoding="utf-8"))
    assert meta["settings"] == snapshot
    assert meta["model_sha256"] == {"depth": "abc"}
    assert meta["input_fingerprint"] == "fp-1"
    assert meta["versions"] == {"v3r": "0.1.0"}
    assert meta["machine"]["python"]


def test_input_fingerprint_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    """The fingerprint is stable for one file and changes when the bytes change."""
    first = tmp_path / "video.mp4"
    second = tmp_path / "video2.mp4"
    first.write_bytes(b"abc")
    second.write_bytes(b"abd")
    assert input_fingerprint(first, None) == input_fingerprint(first, None)
    assert input_fingerprint(first, None) != input_fingerprint(second, None)
