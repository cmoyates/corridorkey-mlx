"""Video inference pipeline for CorridorKey MLX.

Processes frame-aligned input + hint MP4s through GreenFormer,
yielding per-frame results as a generator. Designed for temporal
optimization experiments (V0 baseline, V2 backbone skip, etc.).

VideoProcessor is stateful and separate from the single-frame engine —
video processing has cache, scheduling, and memory management concerns
that don't belong in the stateless engine API.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import numpy as np
from PIL import Image

from corridorkey_mlx.io.preprocess_mlx import preprocess_mlx

if TYPE_CHECKING:
    from collections.abc import Generator

    from corridorkey_mlx.model.corridorkey import GreenFormer

logger = logging.getLogger(__name__)

# Video-specific cache limit — higher than single-frame to allow buffer recycling
# across frames without per-frame gc.collect / mx.clear_cache
VIDEO_CACHE_LIMIT_BYTES = 2 * 1024**3  # 2 GiB


@dataclass
class FrameResult:
    """Per-frame output from VideoProcessor."""

    frame_idx: int
    alpha: np.ndarray  # (H, W) uint8
    fg: np.ndarray  # (H, W, 3) uint8
    decode_time_ms: float  # time to read + resize input + hint
    infer_time_ms: float  # model forward + materialize
    save_time_ms: float  # PNG save (0 if not saving)
    peak_memory_mb: float  # mx.metal.get_peak_memory at this frame
    is_keyframe: bool = True  # True if backbone ran this frame


def _read_video_frame_cv2(cap: Any, frame_idx: int) -> np.ndarray:
    """Read a single frame from cv2.VideoCapture. Returns uint8 RGB HWC."""
    import cv2

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, bgr = cap.read()
    if not ret:
        msg = f"Failed to read frame {frame_idx}"
        raise RuntimeError(msg)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _resize_frame(frame_u8: np.ndarray, target_size: int) -> np.ndarray:
    """Resize uint8 HWC frame to square target_size via PIL bicubic. Returns float32 [0,1]."""
    pil = Image.fromarray(frame_u8)
    if pil.size != (target_size, target_size):
        pil = pil.resize((target_size, target_size), Image.Resampling.BICUBIC)
    return np.asarray(pil, dtype=np.float32) / 255.0


def _save_frame_png(
    alpha_u8: np.ndarray,
    fg_u8: np.ndarray,
    output_dir: Path,
    frame_idx: int,
) -> None:
    """Save alpha + foreground PNGs for a single frame."""
    Image.fromarray(alpha_u8, mode="L").save(output_dir / f"alpha_{frame_idx:04d}.png")
    Image.fromarray(fg_u8, mode="RGB").save(output_dir / f"fg_{frame_idx:04d}.png")


class VideoProcessor:
    """Process frame-aligned input + hint videos through GreenFormer.

    Generator-based: yields FrameResult per frame for progress reporting
    and bounded memory. Does NOT own the model — caller provides it.

    Args:
        model: Loaded GreenFormer (with checkpoint + compilation already done).
        img_size: Model input resolution (square). Frames resized to this.
        output_dir: If provided, save per-frame PNGs (alpha + fg).
        async_save: If True, overlap PNG save with next-frame inference
            via a single-worker ThreadPoolExecutor.
        skip_interval: Backbone skip interval. 1 = run backbone every frame
            (V0 baseline). N > 1 = run backbone every Nth frame, reuse
            cached features for intermediate frames. Frame 0 always runs.
    """

    def __init__(
        self,
        model: GreenFormer,
        img_size: int = 1024,
        output_dir: Path | None = None,
        async_save: bool = True,
        skip_interval: int = 1,
        ema_alpha: float | None = None,
        async_decode: bool = False,
    ) -> None:
        self.model = model
        self.img_size = img_size
        self.output_dir = output_dir
        self.async_save = async_save
        self.skip_interval = max(1, skip_interval)
        self.ema_alpha = ema_alpha  # None = disabled, 0.6-0.8 typical
        self.async_decode = async_decode  # overlap decode N+1 with GPU N

        # Feature cache for backbone skip (bare variables, not a class)
        self._cached_features: list[mx.array] | None = None
        self._cached_frame_idx: int = -1

        # EMA state (output-space blending)
        self._prev_alpha: np.ndarray | None = None
        self._prev_fg: np.ndarray | None = None

        # Set video-appropriate cache limit
        mx.set_cache_limit(VIDEO_CACHE_LIMIT_BYTES)

    def process_video(
        self,
        input_path: str | Path,
        hint_path: str | Path,
    ) -> Generator[FrameResult, None, None]:
        """Process all frames from input + hint videos.

        Args:
            input_path: Path to input RGB video (MP4).
            hint_path: Path to hint/alpha video (MP4). Must have same frame count.

        Yields:
            FrameResult for each frame.

        Raises:
            ValueError: If frame counts don't match.
            FileNotFoundError: If video files don't exist.
        """
        import cv2

        input_path = Path(input_path)
        hint_path = Path(hint_path)

        if not input_path.exists():
            msg = f"Input video not found: {input_path}"
            raise FileNotFoundError(msg)
        if not hint_path.exists():
            msg = f"Hint video not found: {hint_path}"
            raise FileNotFoundError(msg)

        input_cap = cv2.VideoCapture(str(input_path))
        hint_cap = cv2.VideoCapture(str(hint_path))

        try:
            input_frames = int(input_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            hint_frames = int(hint_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Original video dimensions for output resizing
            orig_w = int(input_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(input_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._orig_size: tuple[int, int] = (orig_w, orig_h)  # (W, H) for PIL

            if input_frames != hint_frames:
                msg = (
                    f"Frame count mismatch: input={input_frames}, hint={hint_frames}. "
                    f"Input and hint videos must be frame-aligned."
                )
                raise ValueError(msg)

            if input_frames == 0:
                msg = f"Input video has 0 frames: {input_path}"
                raise ValueError(msg)

            logger.info(
                "Processing %d frames at %dx%d", input_frames, self.img_size, self.img_size
            )

            if self.output_dir is not None:
                self.output_dir.mkdir(parents=True, exist_ok=True)

            yield from self._process_loop(input_cap, hint_cap, input_frames)
        finally:
            input_cap.release()
            hint_cap.release()
            # Clean up after full video — NOT per frame
            mx.clear_cache()

    def _is_keyframe(self, frame_idx: int) -> bool:
        """Whether this frame should run the full backbone."""
        if self.skip_interval <= 1:
            return True
        return frame_idx % self.skip_interval == 0

    def _decode_frame(
        self, input_cap: Any, hint_cap: Any, frame_idx: int
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Decode + resize a single frame pair. Returns (rgb_f32, mask_f32, decode_ms)."""
        t_decode = time.perf_counter()
        rgb_u8 = _read_video_frame_cv2(input_cap, frame_idx)
        hint_u8 = _read_video_frame_cv2(hint_cap, frame_idx)

        rgb_f32 = _resize_frame(rgb_u8, self.img_size)
        hint_gray_u8 = hint_u8[:, :, 2]
        hint_pil = Image.fromarray(hint_gray_u8, mode="L")
        if hint_pil.size != (self.img_size, self.img_size):
            hint_pil = hint_pil.resize(
                (self.img_size, self.img_size), Image.Resampling.BICUBIC
            )
        mask_f32 = np.asarray(hint_pil, dtype=np.float32)[:, :, np.newaxis] / 255.0
        decode_ms = (time.perf_counter() - t_decode) * 1000.0
        return rgb_f32, mask_f32, decode_ms

    def _infer_frame(self, x: mx.array, is_keyframe: bool) -> dict[str, mx.array]:
        """Run inference using decomposed forward with optional backbone skip.

        On keyframes: run_backbone -> cache features -> run_decoders -> run_refiner.
        On non-keyframes: reuse cached features -> run_decoders -> run_refiner.
        """
        if is_keyframe:
            features = self.model.run_backbone(x)
            # CRITICAL: materialize before caching to prevent lazy graph
            # from holding all backbone intermediates alive (~2-3x memory bloat).
            # mx.eval is MLX array materialization, not Python eval().
            mx.eval(*features)  # noqa: S307
            self._cached_features = features
        else:
            features = self._cached_features
            assert features is not None, "No cached features — first frame must be keyframe"

        coarse = self.model.run_decoders(features)
        outputs = self.model.run_refiner(x, coarse)
        return outputs

    def _postprocess_frame(
        self,
        outputs: dict[str, mx.array],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert MLX outputs to uint8 numpy with optional EMA blending."""
        alpha_f32 = np.array(outputs["alpha_final"][0, :, :, 0])
        fg_f32 = np.array(outputs["fg_final"][0])

        # EMA temporal blending (output-space)
        if self.ema_alpha is not None and self._prev_alpha is not None:
            a = self.ema_alpha
            alpha_f32 = a * alpha_f32 + (1.0 - a) * self._prev_alpha
            fg_f32 = a * fg_f32 + (1.0 - a) * self._prev_fg

        if self.ema_alpha is not None:
            self._prev_alpha = alpha_f32.copy()
            self._prev_fg = fg_f32.copy()

        alpha_u8 = (np.clip(alpha_f32, 0.0, 1.0) * 255.0).astype(np.uint8)
        fg_u8 = (np.clip(fg_f32, 0.0, 1.0) * 255.0).astype(np.uint8)

        # Resize back to original video dimensions
        if self._orig_size != (self.img_size, self.img_size):
            alpha_pil = Image.fromarray(alpha_u8, mode="L")
            alpha_u8 = np.asarray(
                alpha_pil.resize(self._orig_size, Image.Resampling.LANCZOS)
            )
            fg_pil = Image.fromarray(fg_u8, mode="RGB")
            fg_u8 = np.asarray(
                fg_pil.resize(self._orig_size, Image.Resampling.LANCZOS)
            )

        return alpha_u8, fg_u8

    def _process_loop(
        self,
        input_cap: Any,
        hint_cap: Any,
        num_frames: int,
    ) -> Generator[FrameResult, None, None]:
        """Inner loop: decode -> preprocess -> infer -> postprocess -> yield.

        When async_decode=True, overlaps frame N+1 decode (CPU) with frame N
        GPU inference via threaded decode-ahead + mx.async_eval.
        """
        save_executor = None
        save_future = None

        if self.async_save and self.output_dir is not None:
            save_executor = ThreadPoolExecutor(max_workers=1)

        # Decode-ahead executor (separate from save executor)
        decode_executor = None
        if self.async_decode:
            decode_executor = ThreadPoolExecutor(max_workers=1)

        # Reset state for this video
        self._cached_features = None
        self._cached_frame_idx = -1
        self._prev_alpha = None
        self._prev_fg = None

        try:
            # Pre-decode frame 0 (always synchronous)
            next_decode = self._decode_frame(input_cap, hint_cap, 0)

            for frame_idx in range(num_frames):
                rgb_f32, mask_f32, decode_ms = next_decode

                # --- Preprocess + Infer ---
                t_infer = time.perf_counter()
                x = preprocess_mlx(rgb_f32, mask_f32)
                is_keyframe = self._is_keyframe(frame_idx)
                outputs = self._infer_frame(x, is_keyframe)

                if self.async_decode and frame_idx + 1 < num_frames:
                    # mx.async_eval: MLX array materialization (not Python eval)
                    mx.async_eval(outputs)  # noqa: S307
                    # Decode next frame on CPU thread while GPU computes
                    decode_future = decode_executor.submit(  # type: ignore[union-attr]
                        self._decode_frame, input_cap, hint_cap, frame_idx + 1
                    )
                    # np.array() blocks until GPU done
                    alpha_u8, fg_u8 = self._postprocess_frame(outputs)
                    infer_ms = (time.perf_counter() - t_infer) * 1000.0
                    next_decode = decode_future.result()
                else:
                    # Synchronous: last frame or async disabled
                    # mx.eval: MLX array materialization (not Python eval)
                    mx.eval(outputs)  # noqa: S307
                    infer_ms = (time.perf_counter() - t_infer) * 1000.0
                    alpha_u8, fg_u8 = self._postprocess_frame(outputs)
                    if frame_idx + 1 < num_frames:
                        next_decode = self._decode_frame(
                            input_cap, hint_cap, frame_idx + 1
                        )

                if is_keyframe:
                    self._cached_frame_idx = frame_idx

                del outputs, x
                peak_mb = mx.get_peak_memory() / 1e6

                # --- Save ---
                save_ms = 0.0
                if self.output_dir is not None:
                    if save_future is not None:
                        save_future.result()
                    if save_executor is not None:
                        save_future = save_executor.submit(
                            _save_frame_png, alpha_u8, fg_u8, self.output_dir, frame_idx
                        )
                    else:
                        t_save = time.perf_counter()
                        _save_frame_png(alpha_u8, fg_u8, self.output_dir, frame_idx)
                        save_ms = (time.perf_counter() - t_save) * 1000.0

                if frame_idx > 0 and frame_idx % 10 == 0:
                    import gc

                    gc.collect()

                yield FrameResult(
                    frame_idx=frame_idx,
                    alpha=alpha_u8,
                    fg=fg_u8,
                    decode_time_ms=decode_ms,
                    infer_time_ms=infer_ms,
                    save_time_ms=save_ms,
                    peak_memory_mb=peak_mb,
                    is_keyframe=is_keyframe,
                )
        finally:
            if save_future is not None:
                save_future.result()
            if save_executor is not None:
                save_executor.shutdown(wait=True)
            if decode_executor is not None:
                decode_executor.shutdown(wait=True)
            self._cached_features = None
