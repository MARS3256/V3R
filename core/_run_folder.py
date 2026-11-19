"""Run-folder paths and creation for one V3R run (plan 5.1).

This module is a private helper of core/debugio.py: it is the only place that builds
run-folder paths. It does NOT write artifacts, run the pipeline, or import bpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TELEMETRY_COLUMNS: tuple[str, ...] = tuple(
    "frame_index t_video_s lat lon alt roll pitch yaw h_agl_m footprint_w_m d_since_last_m "
    "blur_score is_keyframe skip_reason depth_ms fit_scale fit_mad_m ground_frac cloud_pts "
    "fuse_ms regions mesh_ms total_ms gpu_mem_mb".split()
)
_TREE: tuple[str, ...] = tuple(
    "ingest/keyframes depth/raw depth/png diff ground cloud/text heightmap/png objects detect graphs".split()
)


class RunFolderExistsError(FileExistsError):
    """Raised when a run folder already exists; run ids must be unique."""


@dataclass(frozen=True)
class RunFolder:
    """Paths for one run folder; every artifact path is derived here and nowhere else."""

    runs_root: Path
    run_id: str
    root: Path

    def telemetry_csv(self) -> Path:
        """Return the telemetry CSV path."""
        return self.root / "telemetry.csv"

    def skipped_csv(self) -> Path:
        """Return the skipped-frames CSV path."""
        return self.root / "ingest" / "skipped.csv"

    def depth_stats_csv(self) -> Path:
        """Return the depth stats CSV path."""
        return self.root / "depth" / "stats.csv"

    def diff_metrics_csv(self) -> Path:
        """Return the frame-diff metrics CSV path."""
        return self.root / "diff" / "metrics.csv"

    def detections_csv(self) -> Path:
        """Return the detection results CSV path."""
        return self.root / "detect" / "detections.csv"

    def keyframe_jpg(self, index: int) -> Path:
        """Return the keyframe JPEG path for one frame index."""
        return self.root / "ingest" / "keyframes" / _frame_name(index, "jpg")

    def depth_raw_npz(self, index: int) -> Path:
        """Return the raw relative-depth NPZ path for one frame index."""
        return self.root / "depth" / "raw" / _frame_name(index, "npz")

    def depth_png(self, index: int) -> Path:
        """Return the depth PNG path for one frame index."""
        return self.root / "depth" / "png" / _frame_name(index, "png")

    def diff_png(self, index: int) -> Path:
        """Return the frame-diff PNG path for one frame index."""
        return self.root / "diff" / _frame_name(index, "png")

    def ground_png(self, index: int) -> Path:
        """Return the ground-mask overlay PNG path for one frame index."""
        return self.root / "ground" / _frame_name(index, "png")

    def cloud_ply(self, index: int) -> Path:
        """Return the per-keyframe point-cloud PLY path for one frame index."""
        return self.root / "cloud" / _frame_name(index, "ply")

    def cloud_text(self, index: int) -> Path:
        """Return the per-keyframe point-cloud text sample path for one frame index."""
        return self.root / "cloud" / "text" / _frame_name(index, "txt")

    def heightmap_npz(self) -> Path:
        """Return the fused height-map NPZ path."""
        return self.root / "heightmap" / "latest.npz"

    def heightmap_png(self, step: int) -> Path:
        """Return the height-map relief PNG path for one fusion step."""
        return self.root / "heightmap" / "png" / f"step_{step:04d}.png"

    def objects_dir(self, step: int) -> Path:
        """Create and return the per-step object artifact directory."""
        directory = self.root / "objects" / f"step_{step:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def meta_json(self) -> Path:
        """Return the run metadata JSON path."""
        return self.root / "meta.json"

    def report_md(self) -> Path:
        """Return the end-of-run report Markdown path."""
        return self.root / "report.md"

    def graphs_dir(self) -> Path:
        """Return the graphs directory path."""
        return self.root / "graphs"


def create_run_folder(runs_root: Path, run_id: str) -> RunFolder:
    """Create a new run folder with the full plan 5.1 tree, refusing an existing folder."""
    root = Path(runs_root) / run_id
    if root.exists():
        raise RunFolderExistsError(f"run folder already exists: {root}")
    for relative in _TREE:
        (root / relative).mkdir(parents=True, exist_ok=True)
    return RunFolder(runs_root=Path(runs_root), run_id=run_id, root=root)


def _frame_name(index: int, extension: str) -> str:
    """Return a zero-padded frame filename with the given extension."""
    return f"frame_{index:06d}.{extension}"
