#!/usr/bin/env python3
"""Video inference with CorridorKey MLX.

Processes frame-aligned input + hint MP4s, outputs PNG sequences + optional MP4.

Usage:
    uv run python scripts/infer_video.py \\
        --input reference/video/Input.mp4 --hint reference/video/hint.mp4
    uv run python scripts/infer_video.py \\
        --input in.mp4 --hint hint.mp4 --output-dir results/ --stitch
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from corridorkey_mlx.inference.pipeline import DEFAULT_CHECKPOINT, load_model
from corridorkey_mlx.inference.video import VideoProcessor

DEFAULT_IMG_SIZE = 1024


def main() -> None:
    parser = argparse.ArgumentParser(description="CorridorKey MLX video inference")
    parser.add_argument("--input", type=Path, required=True, help="Input RGB video (MP4)")
    parser.add_argument("--hint", type=Path, required=True, help="Hint/alpha video (MP4)")
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="MLX safetensors checkpoint"
    )
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE, help="Model resolution")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/video"), help="Output directory"
    )
    parser.add_argument(
        "--skip", type=int, default=1, help="Backbone skip interval (1=every frame, 3=every 3rd)"
    )
    parser.add_argument("--stitch", action="store_true", help="Reassemble PNGs to MP4 via ffmpeg")
    parser.add_argument("--fps", type=float, default=29.97, help="Output FPS for stitching")
    args = parser.parse_args()

    for p, name in [(args.input, "Input"), (args.hint, "Hint"), (args.checkpoint, "Checkpoint")]:
        if not p.exists():
            print(f"{name} not found: {p}")
            raise SystemExit(1)

    print(f"Loading model (img_size={args.img_size})...")
    t0 = time.perf_counter()
    model = load_model(args.checkpoint, img_size=args.img_size, slim=True)
    print(f"  Loaded in {time.perf_counter() - t0:.2f}s")

    processor = VideoProcessor(
        model=model,
        img_size=args.img_size,
        output_dir=args.output_dir,
        async_save=True,
        skip_interval=args.skip,
    )

    skip_label = f", skip={args.skip}" if args.skip > 1 else ""
    print(f"Processing {args.input}{skip_label} ...")
    t_start = time.perf_counter()

    for result in processor.process_video(args.input, args.hint):
        kf = "K" if result.is_keyframe else " "
        print(
            f"  [{kf}] Frame {result.frame_idx:3d}: "
            f"decode={result.decode_time_ms:5.1f}ms  "
            f"infer={result.infer_time_ms:5.1f}ms  "
            f"mem={result.peak_memory_mb:.0f}MB"
        )

    total_s = time.perf_counter() - t_start
    print(f"\nDone: {total_s:.2f}s total")

    if args.stitch:
        _stitch_videos(args.output_dir, args.fps)


def _ffmpeg(cmd: list[str], label: str) -> None:
    """Run ffmpeg, print label, suppress output unless it fails."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ffmpeg error ({label}):\n{result.stderr}")
        raise SystemExit(1)


def _stitch_videos(output_dir: Path, fps: float) -> None:
    """Reassemble PNGs to alpha, foreground, and composite MP4s."""
    fps_str = str(fps)

    # Alpha matte video (grayscale)
    alpha_mp4 = output_dir / "alpha.mp4"
    _ffmpeg(
        [
            "ffmpeg", "-y", "-framerate", fps_str,
            "-i", str(output_dir / "alpha_%04d.png"),
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            str(alpha_mp4),
        ],
        "alpha",
    )
    print(f"  Saved: {alpha_mp4}")

    # Foreground video (RGB)
    fg_mp4 = output_dir / "fg.mp4"
    _ffmpeg(
        [
            "ffmpeg", "-y", "-framerate", fps_str,
            "-i", str(output_dir / "fg_%04d.png"),
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            str(fg_mp4),
        ],
        "fg",
    )
    print(f"  Saved: {fg_mp4}")

    # Side-by-side composite: [alpha | foreground]
    comp_mp4 = output_dir / "comp.mp4"
    _ffmpeg(
        [
            "ffmpeg", "-y",
            "-i", str(alpha_mp4), "-i", str(fg_mp4),
            "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
            "-map", "[v]",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            str(comp_mp4),
        ],
        "comp",
    )
    print(f"  Saved: {comp_mp4}")


if __name__ == "__main__":
    main()
