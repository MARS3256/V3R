"""Run-folder artifacts: CSV appenders, meta.json and the PNG write gate (plan 5.1).

This module does NOT run the pipeline or import bpy. It is the public import surface for
run-folder paths; path construction lives in the private core._run_folder helper so this
module stays within the 200-line rule (R3).
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from core._run_folder import (
    TELEMETRY_COLUMNS,
    RunFolder,
    RunFolderExistsError,
    create_run_folder,
)

__all__ = [
    "TELEMETRY_COLUMNS",
    "CsvAppender",
    "RunFolder",
    "RunFolderExistsError",
    "create_run_folder",
    "input_fingerprint",
    "png_gate",
    "write_meta",
]

_FINGERPRINT_BLOCK = 1024 * 1024


class CsvAppender:
    """Append-only CSV writer that writes the header once and flushes every row."""

    def __init__(self, path: Path, columns: Sequence[str]) -> None:
        """Open the CSV, writing the header only when the file is new or empty."""
        self.path = Path(path)
        self.columns = tuple(columns)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        self._handle = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle, lineterminator="\n")
        if is_new:
            self._writer.writerow(self.columns)
            self._handle.flush()

    def append_row(self, values: Sequence[object]) -> None:
        """Append one row and flush it to disk."""
        if len(values) != len(self.columns):
            raise ValueError(f"row has {len(values)} values, expected {len(self.columns)}")
        self._writer.writerow(values)
        self._handle.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        self._handle.close()


def input_fingerprint(video: Path, telemetry: Path | None) -> str:
    """Return a sha256 identity fingerprint of the input files, not a security check."""
    digest = hashlib.sha256()
    for path in (video, telemetry):
        if path is not None:
            digest.update(_fingerprint_file(Path(path)))
    return digest.hexdigest()


def write_meta(
    run: RunFolder,
    settings_snapshot: dict[str, object],
    model_hashes: dict[str, str],
    input_fingerprint: str,
    versions: dict[str, str],
) -> Path:
    """Write meta.json with settings, model hashes, input fingerprint, versions and machine info."""
    meta = {
        "run_id": run.run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "settings": settings_snapshot,
        "model_sha256": model_hashes,
        "input_fingerprint": input_fingerprint,
        "versions": versions,
        "machine": _machine_info(),
    }
    run.meta_json().write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return run.meta_json()


def png_gate(run: RunFolder, debug_mode: str, fingerprint: str) -> bool:
    """Return True when PNG artifacts should be written for this run (plan 5.5)."""
    if debug_mode == "on":
        return True
    if debug_mode == "off":
        return False
    if debug_mode != "always_first_run":
        raise ValueError(f"unknown debug_artifacts mode: {debug_mode}")
    return not _sibling_has_fingerprint(run, fingerprint)


def _fingerprint_file(path: Path) -> bytes:
    """Return file size plus the first and last 1 MB of a file as bytes."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        head = handle.read(_FINGERPRINT_BLOCK)
        if size > _FINGERPRINT_BLOCK:
            handle.seek(-_FINGERPRINT_BLOCK, 2)
            tail = handle.read(_FINGERPRINT_BLOCK)
        else:
            tail = b""
    return str(size).encode("ascii") + head + tail


def _machine_info() -> dict[str, str]:
    """Return a small machine description for meta.json."""
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def _sibling_has_fingerprint(run: RunFolder, fingerprint: str) -> bool:
    """Return True when another run folder records the same input fingerprint."""
    for sibling in run.runs_root.iterdir():
        if not sibling.is_dir() or sibling == run.root:
            continue
        try:
            meta = json.loads((sibling / "meta.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("input_fingerprint") == fingerprint:
            return True
    return False
