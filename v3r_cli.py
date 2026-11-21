#!/usr/bin/env python3
"""V3R CLI: run Solv3R streaming reconstruction from a video file or live stream.

This is the command-line twin of the Blender worker. It writes the same samples/outputs tree:
depth NPZ/PNG, per-frame points, a cumulative NPZ/PLY and report.md.

Examples:
  .venv-worker/bin/python v3r_cli.py --source samples/drone1.mp4 --sample-fps 2
  .venv-worker/bin/python v3r_cli.py --source rtmp://localhost/live/stream --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from worker.app import Worker
from worker.config import WorkerConfig

_STREAM_PREFIXES = ("rtsp://", "rtmp://", "udp://", "srt://", "http://", "https://")


def _source_kind(source: str, override: str) -> str:
    """Return video or stream, auto-detecting from the URL scheme when asked."""
    if override != "auto":
        return override
    return "stream" if source.lower().startswith(_STREAM_PREFIXES) else "video"


def build_config(args: argparse.Namespace) -> WorkerConfig:
    """Build a validated WorkerConfig from CLI arguments."""
    return WorkerConfig(
        source_kind=_source_kind(args.source, args.source_kind),
        source=args.source,
        out_dir=str(Path(args.out).expanduser().resolve()),
        checkpoint=str(Path(args.checkpoint).expanduser().resolve()),
        device=args.device,
        preset=args.preset,
        image_long_side=args.image_long_side,
        sample_fps=args.sample_fps,
        max_frames=args.max_frames,
        point_stride=args.point_stride,
        point_budget=args.point_budget,
        depth_enabled=not args.no_depth,
        stream_lost_timeout_s=args.stream_lost_timeout,
        anchors=args.anchors,
        recent=args.recent,
        memory=args.memory,
        memory_stride=args.memory_stride,
        calls=args.calls,
        anchor_calls=args.anchor_calls,
        refresh_tau=args.refresh_tau,
        refresh_check=args.refresh_check,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="video file path or stream URL")
    parser.add_argument("--source-kind", default="auto", choices=("auto", "video", "stream"))
    parser.add_argument("--out", default="samples/outputs", help="output directory")
    parser.add_argument("--checkpoint", default="models/solv3r.safetensors", help="local .safetensors file")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--preset", default="S4", choices=("S4", "L4", "L8"))
    parser.add_argument("--image-long-side", type=int, default=252, choices=(252, 504))
    parser.add_argument("--sample-fps", type=float, default=2.0, help="0 processes every frame")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means no limit")
    parser.add_argument("--point-stride", type=int, default=3)
    parser.add_argument("--point-budget", type=int, default=500_000)
    parser.add_argument("--no-depth", action="store_true", help="skip depth NPZ/PNG output")
    parser.add_argument("--stream-lost-timeout", type=float, default=10.0)
    parser.add_argument("--anchors", type=int, default=4)
    parser.add_argument("--recent", type=int, default=4)
    parser.add_argument("--memory", type=int, default=16)
    parser.add_argument("--memory-stride", type=int, default=16)
    parser.add_argument("--calls", type=int, default=4)
    parser.add_argument("--anchor-calls", type=int, default=16)
    parser.add_argument("--refresh-tau", type=float, default=0.02)
    parser.add_argument("--refresh-check", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build the config, run the worker and print a human summary to stderr."""
    args = parse_args(argv)
    cfg = build_config(args)
    print(f"V3R source: {cfg.source_kind} {cfg.source}", file=sys.stderr)
    print(f"V3R device: {cfg.device}, preset {cfg.preset}, {cfg.image_long_side} px", file=sys.stderr)
    print(f"V3R outputs: {cfg.out_dir}", file=sys.stderr)
    code = Worker(cfg).run()
    print(f"V3R exit code: {code}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
