"""Frame source for video files and live streams, decoded with OpenCV/FFmpeg.

This module does NOT run inference or write outputs. It yields RGB frames at the configured
sample rate and raises SourceError / StreamLostError with the path or URL when input fails.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from worker.config import WorkerConfig


class SourceError(RuntimeError):
    """Raised when a video file or stream cannot be opened or decoded."""


class StreamLostError(SourceError):
    """Raised when a live stream stops producing frames for longer than the configured timeout."""


@dataclass(frozen=True)
class Frame:
    """One decoded frame: its source index, timestamp and RGB image."""

    index: int
    t_s: float
    rgb: np.ndarray


class FrameSource:
    """Iterate frames from a video file or a live stream at the configured sample rate."""

    def __init__(self, cfg: WorkerConfig) -> None:
        """Open the source and read its frame rate; raise SourceError when it cannot open."""
        self.cfg = cfg
        self.is_stream = cfg.source_kind == "stream"
        self.cap = cv2.VideoCapture(cfg.source, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise SourceError(f"cannot open {cfg.source_kind} {cfg.source!r} with OpenCV/FFmpeg")
        if self.is_stream:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            for name, value in (("CAP_PROP_OPEN_TIMEOUT_MSEC", 8000), ("CAP_PROP_READ_TIMEOUT_MSEC", 8000)):
                prop = getattr(cv2, name, None)
                if prop is not None:
                    self.cap.set(prop, value)
        self.src_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.step = 1
        if not self.is_stream and cfg.sample_fps > 0:
            if not np.isfinite(self.src_fps) or self.src_fps <= 0:
                raise SourceError(f"video {cfg.source!r} reports no usable frame rate (got {self.src_fps!r})")
            self.step = max(1, round(self.src_fps / cfg.sample_fps))
        self.start_time = time.monotonic()
        self.last_frame_time = self.start_time
        self.last_yield = 0.0
        self.yielded = 0
        self.read_index = 0

    def __iter__(self) -> Iterator[Frame]:
        """Yield frames until the file ends, the frame budget is reached or the stream is lost."""
        while self.yielded < self.cfg.max_frames or self.cfg.max_frames == 0:
            ok, frame = self.cap.read()
            now = time.monotonic()
            if not ok:
                if not self.is_stream:
                    return
                if now - self.last_frame_time > self.cfg.stream_lost_timeout_s:
                    raise StreamLostError(
                        f"stream {self.cfg.source!r} produced no frame for "
                        f"{now - self.last_frame_time:.1f}s (timeout {self.cfg.stream_lost_timeout_s:.1f}s)"
                    )
                time.sleep(0.01)
                continue
            self.last_frame_time = now
            self.read_index += 1
            if not self.is_stream and (self.read_index - 1) % self.step != 0:
                continue
            if self.is_stream and self.cfg.sample_fps > 0 and now - self.last_yield < 1.0 / self.cfg.sample_fps:
                continue
            self.last_yield = now
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield Frame(index=self.yielded, t_s=now - self.start_time, rgb=np.ascontiguousarray(rgb))
            self.yielded += 1

    def close(self) -> None:
        """Release the OpenCV capture."""
        self.cap.release()
