#!/usr/bin/env python3
"""Experiment: selective refinement vs full-frame baseline.

Runs both inference modes on the same image, compares timing and quality,
saves artifact images and a text report.

WARNING: Experimental — not part of the stable API.

Usage:
    uv run python scripts/experiment_selective_refine.py \
        --image samples/sample.png \
        --hint samples/hint.png \
        --checkpoint checkpoints/corridorkey_mlx.safetensors \
        --output-dir output/selective_refine/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Ensure package importable from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from corridorkey_mlx import CorridorKeyMLXEngine
from corridorkey_mlx.inference.pipeline import load_model
from corridorkey_mlx.inference.selective_refine import (
    SelectiveRefineConfig,
    SelectiveRefineResult,
    selective_refine,
)

DEFAULT_CHECKPOINT = Path("checkpoints/corridorkey_mlx.safetensors")
DEFAULT_OUTPUT_DIR = Path("output/selective_refine")
DEFAULT_COARSE_SIZE = 512
DEFAULT_TILE_SIZE = 512
DEFAULT_TILE_OVERLAP = 64


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def load_inputs(image_path: Path, hint_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load RGB image and alpha hint as uint8 arrays."""
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    hint = np.asarray(Image.open(hint_path).convert("L"))
    return rgb, hint


# ---------------------------------------------------------------------------
# Baseline inference (full-frame via engine)
# ---------------------------------------------------------------------------


def run_baseline(
    checkpoint: Path,
    rgb: np.ndarray,
    hint: np.ndarray,
    img_size: int,
    compile_model: bool,
) -> tuple[dict[str, np.ndarray], float]:
    """Run full-frame baseline via CorridorKeyMLXEngine.

    Returns (result_dict, elapsed_ms).
    """
    engine = CorridorKeyMLXEngine(
        checkpoint_path=checkpoint,
        img_size=img_size,
        compile=compile_model,
    )

    # Warmup
    _ = engine.process_frame(rgb, hint)

    start = time.perf_counter()
    result = engine.process_frame(rgb, hint)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


# ---------------------------------------------------------------------------
# Selective refinement
# ---------------------------------------------------------------------------


def run_selective(
    checkpoint: Path,
    rgb: np.ndarray,
    hint: np.ndarray,
    config: SelectiveRefineConfig,
) -> tuple[SelectiveRefineResult, float]:
    """Run selective refinement pipeline.

    Returns (SelectiveRefineResult, total_elapsed_ms).
    """
    model = load_model(
        checkpoint,
        img_size=config.tile_size,
        compile=config.compile,
    )

    # Warmup (coarse-sized input)
    dummy_h = dummy_w = config.coarse_size
    dummy_rgb = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)
    dummy_hint = np.zeros((dummy_h, dummy_w), dtype=np.uint8)
    _ = selective_refine(model, dummy_rgb, dummy_hint, config)

    start = time.perf_counter()
    result = selective_refine(model, rgb, hint, config)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


# ---------------------------------------------------------------------------
# Quality comparison
# ---------------------------------------------------------------------------


def compute_diff_stats(baseline: np.ndarray, selective: np.ndarray) -> dict[str, float]:
    """Compute absolute difference statistics between two uint8 arrays."""
    diff = np.abs(baseline.astype(np.float32) - selective.astype(np.float32)) / 255.0
    return {
        "max_abs_diff": float(np.max(diff)),
        "mean_abs_diff": float(np.mean(diff)),
        "median_abs_diff": float(np.median(diff)),
        "p95_abs_diff": float(np.percentile(diff, 95)),
        "p99_abs_diff": float(np.percentile(diff, 99)),
    }


def make_diff_heatmap(
    baseline: np.ndarray, selective: np.ndarray, scale: float = 10.0
) -> np.ndarray:
    """Create a heatmap of |baseline - selective| scaled for visibility.

    Returns (H, W, 3) uint8 RGB.
    """
    if baseline.ndim == 2:
        diff = np.abs(baseline.astype(np.float32) - selective.astype(np.float32))
    else:
        diff = np.mean(
            np.abs(baseline.astype(np.float32) - selective.astype(np.float32)),
            axis=-1,
        )
    # Scale up for visibility
    heatmap = np.clip(diff * scale, 0, 255).astype(np.uint8)
    # Colorize: black -> red -> yellow
    r = np.clip(heatmap * 2, 0, 255).astype(np.uint8)
    g = np.clip(heatmap * 2 - 255, 0, 255).astype(np.uint8)
    b = np.zeros_like(heatmap)
    return np.stack([r, g, b], axis=-1)


# ---------------------------------------------------------------------------
# Tile overlay visualization
# ---------------------------------------------------------------------------


def draw_tile_overlay(
    rgb: np.ndarray,
    tile_coords: list[tuple[int, int, int, int]],
    mask: np.ndarray,
) -> np.ndarray:
    """Draw tile rectangles and uncertainty mask overlay on the source image.

    Returns (H, W, 3) uint8.
    """
    img = Image.fromarray(rgb).convert("RGBA")

    # Semi-transparent red overlay for uncertainty mask
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    mask_u8 = (mask.astype(np.float32) * 80).clip(0, 255).astype(np.uint8)
    red_channel = np.zeros((*mask.shape, 4), dtype=np.uint8)
    red_channel[:, :, 0] = 255
    red_channel[:, :, 3] = mask_u8
    overlay = Image.fromarray(red_channel, mode="RGBA")
    img = Image.alpha_composite(img, overlay)

    # Draw tile rectangles
    draw = ImageDraw.Draw(img)
    for y0, x0, y1, x1 in tile_coords:
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(0, 255, 0), width=2)

    return np.asarray(img.convert("RGB"))


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    image_shape: tuple[int, ...],
    config: SelectiveRefineConfig,
    baseline_ms: float,
    sr_result: SelectiveRefineResult,
    selective_ms: float,
    alpha_diff: dict[str, float],
    fg_diff: dict[str, float],
) -> str:
    """Generate text report comparing baseline and selective refinement."""
    h, w = image_shape[:2]
    stats = sr_result.stats

    tiles_selected = stats.get("tiles_selected", 0)
    tiles_total = stats.get("tiles_total", 0)
    tile_pct = (tiles_selected / max(tiles_total, 1)) * 100
    mask_coverage = stats.get("mask_coverage", 0.0)

    speedup = baseline_ms / max(selective_ms, 0.1)

    lines = [
        "=== Selective Refinement Experiment ===",
        f"Image: {w}x{h}",
        f"Coarse size: {config.coarse_size}",
        f"Tile size: {config.tile_size}, overlap: {config.tile_overlap}",
        f"Uncertainty band: [{config.uncertainty_low}, {config.uncertainty_high}]",
        f"Gradient threshold: {config.gradient_threshold}",
        f"Dilation radius: {config.dilation_radius}",
        f"Min tile coverage: {config.min_tile_coverage}",
        "",
        "Baseline:",
        f"  Time: {baseline_ms:.0f} ms",
        "",
        "Selective:",
        f"  Coarse pass: {stats.get('coarse_ms', 0):.0f} ms",
        f"  Mask coverage: {mask_coverage * 100:.1f}%",
        f"  Tiles selected: {tiles_selected} / {tiles_total} ({tile_pct:.1f}% of frame)",
        f"  Tile refinement: {stats.get('tile_refine_ms', 0):.0f} ms",
        f"  Stitching: {stats.get('stitch_ms', 0):.0f} ms",
        f"  Total: {selective_ms:.0f} ms",
        f"  Speedup: {speedup:.2f}x",
        "",
        "Quality (vs baseline):",
        f"  Alpha max_abs_diff: {alpha_diff['max_abs_diff']:.4f}",
        f"  Alpha mean_abs_diff: {alpha_diff['mean_abs_diff']:.6f}",
        f"  Alpha p99_abs_diff: {alpha_diff['p99_abs_diff']:.6f}",
        f"  FG max_abs_diff: {fg_diff['max_abs_diff']:.4f}",
        f"  FG mean_abs_diff: {fg_diff['mean_abs_diff']:.6f}",
        f"  FG p99_abs_diff: {fg_diff['p99_abs_diff']:.6f}",
        "",
    ]

    # Warnings
    warnings = []
    if mask_coverage > 0.90:
        warnings.append("High mask coverage (>90%) — limited speedup expected")
    if mask_coverage < 0.01:
        warnings.append("Very low mask coverage (<1%) — coarse may suffice")
    if alpha_diff["max_abs_diff"] > 0.05:
        warnings.append(
            f"Alpha max_abs_diff ({alpha_diff['max_abs_diff']:.4f}) exceeds 0.05 threshold"
        )
    if speedup < 1.0:
        warnings.append(f"Selective is slower than baseline ({speedup:.2f}x)")

    if warnings:
        lines.append("Warnings:")
        for w_msg in warnings:
            lines.append(f"  - {w_msg}")
    else:
        lines.append("Warnings: none")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment: selective refinement vs baseline")
    parser.add_argument("--image", type=Path, required=True, help="RGB input image")
    parser.add_argument("--hint", type=Path, required=True, help="Grayscale alpha hint")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="MLX safetensors checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output artifacts",
    )
    parser.add_argument(
        "--coarse-size",
        type=int,
        default=DEFAULT_COARSE_SIZE,
        help="Coarse pass resolution",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help="Tile size (must match model img_size)",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=DEFAULT_TILE_OVERLAP,
        help="Overlap between tiles",
    )
    parser.add_argument(
        "--uncertainty-low",
        type=float,
        default=0.05,
        help="Lower bound of uncertainty band",
    )
    parser.add_argument(
        "--uncertainty-high",
        type=float,
        default=0.95,
        help="Upper bound of uncertainty band",
    )
    parser.add_argument(
        "--gradient-threshold",
        type=float,
        default=0.02,
        help="Sobel gradient threshold for edges",
    )
    parser.add_argument(
        "--dilation-radius",
        type=int,
        default=32,
        help="Dilation radius for uncertainty mask",
    )
    parser.add_argument(
        "--min-tile-coverage",
        type=float,
        default=0.05,
        help="Min fraction of uncertain pixels to keep a tile",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable mx.compile",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline run (selective only)",
    )
    parser.add_argument(
        "--baseline-img-size",
        type=int,
        default=None,
        help="Baseline model resolution (default: same as input image size)",
    )
    args = parser.parse_args()

    # -- Validate inputs --
    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("Run scripts/convert_weights.py first.")
        sys.exit(1)
    if not args.image.exists():
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)
    if not args.hint.exists():
        print(f"ERROR: Hint not found: {args.hint}")
        sys.exit(1)

    # -- Load inputs --
    print(f"Loading inputs: image={args.image}, hint={args.hint}")
    rgb, hint = load_inputs(args.image, args.hint)
    full_h, full_w = rgb.shape[:2]
    print(f"Input: {full_w}x{full_h}")

    use_compile = not args.no_compile

    # -- Build config --
    config = SelectiveRefineConfig(
        coarse_size=args.coarse_size,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        uncertainty_low=args.uncertainty_low,
        uncertainty_high=args.uncertainty_high,
        gradient_threshold=args.gradient_threshold,
        dilation_radius=args.dilation_radius,
        min_tile_coverage=args.min_tile_coverage,
        compile=use_compile,
    )

    # -- Output dir --
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # -- Baseline --
    baseline_ms = 0.0
    baseline_result: dict[str, np.ndarray] | None = None

    if not args.skip_baseline:
        baseline_img_size = args.baseline_img_size or max(full_h, full_w)
        print(f"\nRunning baseline (img_size={baseline_img_size}, compile={use_compile})...")
        baseline_result, baseline_ms = run_baseline(
            args.checkpoint, rgb, hint, baseline_img_size, use_compile
        )
        print(f"  Baseline: {baseline_ms:.0f} ms")

        Image.fromarray(baseline_result["alpha"], mode="L").save(out / "baseline_alpha.png")
        Image.fromarray(baseline_result["fg"], mode="RGB").save(out / "baseline_fg.png")

    # -- Selective refinement --
    print(
        f"\nRunning selective refinement (coarse={config.coarse_size}, tile={config.tile_size})..."
    )
    sr_result, selective_ms = run_selective(args.checkpoint, rgb, hint, config)
    print(f"  Selective: {selective_ms:.0f} ms")
    tiles_sel = sr_result.stats.get("tiles_selected", 0)
    tiles_tot = sr_result.stats.get("tiles_total", 0)
    print(f"  Tiles: {tiles_sel} / {tiles_tot}")

    # -- Save selective outputs --
    Image.fromarray(sr_result.alpha_final, mode="L").save(out / "selective_alpha.png")
    Image.fromarray(sr_result.fg_final, mode="RGB").save(out / "selective_fg.png")
    Image.fromarray(sr_result.coarse_alpha, mode="L").save(out / "coarse_alpha.png")

    # Uncertainty mask
    mask_vis = sr_result.uncertainty_mask.astype(np.uint8) * 255
    Image.fromarray(mask_vis, mode="L").save(out / "uncertainty_mask.png")

    # Tile overlay
    tile_overlay = draw_tile_overlay(rgb, sr_result.tile_coords, sr_result.uncertainty_mask)
    Image.fromarray(tile_overlay, mode="RGB").save(out / "tile_overlay.png")

    # -- Quality comparison --
    alpha_diff: dict[str, float] = {}
    fg_diff: dict[str, float] = {}

    if baseline_result is not None:
        # Resize selective output to match baseline if needed
        sel_alpha = sr_result.alpha_final
        sel_fg = sr_result.fg_final
        base_alpha = baseline_result["alpha"]
        base_fg = baseline_result["fg"]

        # Ensure same shape
        if sel_alpha.shape != base_alpha.shape:
            bh, bw = base_alpha.shape[:2]
            sel_alpha = np.asarray(
                Image.fromarray(sel_alpha, mode="L").resize((bw, bh), Image.BICUBIC),
                dtype=np.uint8,
            )
            sel_fg = np.asarray(
                Image.fromarray(sel_fg, mode="RGB").resize((bw, bh), Image.BICUBIC),
                dtype=np.uint8,
            )

        alpha_diff = compute_diff_stats(base_alpha, sel_alpha)
        fg_diff = compute_diff_stats(base_fg, sel_fg)

        # Diff heatmap
        diff_heatmap = make_diff_heatmap(base_alpha, sel_alpha)
        Image.fromarray(diff_heatmap, mode="RGB").save(out / "diff_alpha.png")

        print(f"\n  Alpha max_abs_diff: {alpha_diff['max_abs_diff']:.4f}")
        print(f"  Alpha mean_abs_diff: {alpha_diff['mean_abs_diff']:.6f}")
        print(f"  FG max_abs_diff: {fg_diff['max_abs_diff']:.4f}")

        if baseline_ms > 0:
            speedup = baseline_ms / max(selective_ms, 0.1)
            print(f"  Speedup: {speedup:.2f}x")

    # -- Report --
    if baseline_result is not None:
        report = generate_report(
            rgb.shape,
            config,
            baseline_ms,
            sr_result,
            selective_ms,
            alpha_diff,
            fg_diff,
        )
        report_path = out / "report.txt"
        report_path.write_text(report)
        print(f"\nReport saved to {report_path}")
        print()
        print(report)
    else:
        # Selective-only summary
        stats = sr_result.stats
        print("\nSelective-only summary:")
        print(f"  Coarse: {stats.get('coarse_ms', 0):.0f} ms")
        print(f"  Tile refinement: {stats.get('tile_refine_ms', 0):.0f} ms")
        print(f"  Stitching: {stats.get('stitch_ms', 0):.0f} ms")
        print(f"  Total: {selective_ms:.0f} ms")

    print(f"\nArtifacts saved to {out}/")


if __name__ == "__main__":
    main()
