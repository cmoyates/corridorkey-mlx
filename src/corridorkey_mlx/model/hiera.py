"""Hiera backbone — MLX port.

Hierarchical vision transformer (timm hiera_base_plus_224) for feature extraction.
Emits 4 multiscale feature maps in NHWC format.

Ported from: timm/models/hiera.py (Meta Platforms, Apache-2.0 / CC-BY-NC-4.0)
Reference: https://arxiv.org/abs/2306.00989
"""

from __future__ import annotations

import math
from functools import reduce
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open

if TYPE_CHECKING:
    from pathlib import Path

# ── hiera_base_plus_224 constants ──────────────────────────────────────
EMBED_DIM = 112
NUM_HEADS = 2
STAGES = (2, 3, 16, 3)
Q_POOL = 3  # number of stages with q-pooling
Q_STRIDE = (2, 2)
MASK_UNIT_SIZE = (8, 8)
MASK_UNIT_ATTN = (True, True, False, False)
PATCH_KERNEL = (7, 7)
PATCH_STRIDE = (4, 4)
PATCH_PADDING = (3, 3)
IN_CHANS = 4  # RGB + alpha hint
MLP_RATIO = 4.0
DIM_MUL = 2.0
HEAD_MUL = 2.0
ENCODER_KEY_PREFIX = "encoder.model."
TRAIN_IMG_SIZE = 2048  # checkpoint was trained at this resolution


# ── Helpers ────────────────────────────────────────────────────────────


def _prod(seq: tuple[int, ...] | list[int]) -> int:
    return reduce(lambda a, b: a * b, seq, 1)


def undo_windowing(
    x: mx.array,
    shape: list[int],
    mu_shape: list[int],
) -> mx.array:
    """Undo mask-unit windowing: [B, #MUs, MUy, MUx, C] -> [B, H, W, C].

    Faithful port of timm ``undo_windowing`` (2d only).
    """
    ndim = len(shape)  # spatial dims (2 for images)
    batch_size = x.shape[0]
    channels = x.shape[-1]

    num_mus = [s // mu for s, mu in zip(shape, mu_shape, strict=True)]
    # [B, #MUy*#MUx, MUy, MUx, C] -> [B, #MUy, #MUx, MUy, MUx, C]
    x = x.reshape([batch_size] + num_mus + mu_shape + [channels])

    # Interleave: [B, #MUy, MUy, #MUx, MUx, C]
    perm = (
        [0]
        + sum(
            [list(p) for p in zip(range(1, 1 + ndim), range(1 + ndim, 1 + 2 * ndim), strict=True)],
            [],
        )
        + [len(x.shape) - 1]
    )
    x = mx.transpose(x, axes=perm)
    return x.reshape([batch_size] + shape + [channels])


def unroll(x: mx.array, spatial_size: list[int], schedule: list[tuple[int, int]]) -> mx.array:
    """Reorder tokens so patches are contiguous for windowed ops.

    Faithful port of timm ``Unroll.forward`` (2d only, inference path).

    Input:  [B, N, C] (flattened patch embeddings)
    Output: [B', N', C] where B' = B * prod(all strides), N' = prod(cur_size)
    """
    batch_size = x.shape[0]
    channels = x.shape[-1]
    cur_size = list(spatial_size)

    # [B, N, C] -> [B, H, W, C]
    x = x.reshape([batch_size] + cur_size + [channels])

    for strides in schedule:
        cur_size = [i // s for i, s in zip(cur_size, strides, strict=True)]
        # [B, H//Sy, Sy, W//Sx, Sx, C]
        pairs = sum([[i, s] for i, s in zip(cur_size, strides, strict=True)], [])
        new_shape = [batch_size] + pairs + [channels]
        x = x.reshape(new_shape)

        # [B, Sy, Sx, H//Sy, W//Sx, C]
        ndims = len(new_shape)
        perm = [0] + list(range(2, ndims - 1, 2)) + list(range(1, ndims - 1, 2)) + [ndims - 1]
        x = mx.transpose(x, axes=perm)

        # Flatten strides into batch: B' = B * Sy * Sx
        stride_count = _prod(strides)
        x = x.reshape([batch_size * stride_count] + cur_size + [channels])
        batch_size *= stride_count

    # Flatten spatial back using original size → [B, N_orig, C]
    # This collapses the inflated batch back to original B
    return x.reshape(-1, _prod(spatial_size), channels)


def reroll(
    x: mx.array,
    block_idx: int,
    schedule_map: dict[int, tuple[list[tuple[int, int]], list[int]]],
) -> mx.array:
    """Undo unroll to recover spatial layout.

    Faithful port of timm ``Reroll.forward`` (2d only, no-mask inference path).

    Input:  [B', N, C]
    Output: [B, H, W, C] (NHWC)
    """
    remaining_schedule, size = schedule_map[block_idx]
    batch_size = x.shape[0]
    num_tokens = x.shape[1]
    channels = x.shape[-1]

    ndim = len(size)  # 2 for images
    cur_mu_shape = [1] * ndim

    for strides in remaining_schedule:
        # [B, *strides, N//(Sy*Sx), *cur_mu_shape, C]
        stride_prod = _prod(strides)
        inner_n = num_tokens // stride_prod
        x = x.reshape([batch_size] + list(strides) + [inner_n] + cur_mu_shape + [channels])

        # Permute: [B, N//(Sy*Sx), Sy, MUy, Sx, MUx, C]
        total_dims = len(x.shape)
        perm = (
            [0, 1 + ndim]
            + sum(
                [
                    list(p)
                    for p in zip(
                        range(1, 1 + ndim), range(1 + ndim + 1, total_dims - 1), strict=True
                    )
                ],
                [],
            )
            + [total_dims - 1]
        )
        x = mx.transpose(x, axes=perm)

        # Update mu_shape and reshape
        for i in range(ndim):
            cur_mu_shape[i] *= strides[i]
        x = x.reshape([batch_size, -1] + cur_mu_shape + [channels])
        num_tokens = x.shape[1]

    # [B, #MUs, MUy, MUx, C]
    x = x.reshape([batch_size, num_tokens] + cur_mu_shape + [channels])

    # No mask -> return [B, H, W, C]
    return undo_windowing(x, size, cur_mu_shape)


def _interpolate_pos_embed(
    ckpt_embed: mx.array,
    target_tokens: int,
) -> mx.array:
    """Bicubic interpolation of pos_embed from checkpoint to model resolution.

    Args:
        ckpt_embed: (1, N_ckpt, C) from checkpoint
        target_tokens: target token count N_model = H_model * W_model

    Returns:
        (1, N_model, C)
    """
    ckpt_n = ckpt_embed.shape[1]
    if ckpt_n == target_tokens:
        return ckpt_embed

    embed_dim = ckpt_embed.shape[2]
    ckpt_side = int(math.sqrt(ckpt_n))
    model_side = int(math.sqrt(target_tokens))

    # (1, N, C) -> (1, H, W, C) NHWC for MLX upsample
    embed = ckpt_embed.reshape(1, ckpt_side, ckpt_side, embed_dim)
    scale = model_side / ckpt_side
    resizer = nn.Upsample(scale_factor=(scale, scale), mode="cubic", align_corners=False)
    embed = resizer(embed)
    # Back to (1, N, C)
    return embed.reshape(1, target_tokens, embed_dim)


# ── Modules ────────────────────────────────────────────────────────────


class HieraPatchEmbed(nn.Module):
    """Patch embedding: Conv2d(4->112, 7x7, stride=4) + flatten to [B, N, C]."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            IN_CHANS,
            EMBED_DIM,
            kernel_size=PATCH_KERNEL[0],
            stride=PATCH_STRIDE[0],
            padding=PATCH_PADDING[0],
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Input: (B, H, W, 4) NHWC. Output: (B, N, C) where N = (H/4)*(W/4)."""
        x = self.proj(x)  # (B, H/4, W/4, 112)
        batch_size = x.shape[0]
        channels = x.shape[-1]
        return x.reshape(batch_size, -1, channels)  # (B, N, 112)


class HieraMLP(nn.Module):
    """MLP: fc1(dim -> 4*dim) -> GELU -> fc2(4*dim -> dim)."""

    def __init__(self, dim: int, mlp_ratio: float = MLP_RATIO) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class MaskUnitAttention(nn.Module):
    """Windowed multi-head attention with optional q max-pool.

    Operates on unrolled [B, N, C] tokens. When use_mask_unit_attn=True,
    attention is computed within windows of size window_size.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
    ) -> None:
        super().__init__()
        self.dim_out = dim_out
        self.heads = heads
        self.q_stride = q_stride
        self.head_dim = dim_out // heads
        self.scale = self.head_dim**-0.5
        self.window_size = window_size
        self.use_mask_unit_attn = use_mask_unit_attn

        self.qkv = nn.Linear(dim, 3 * dim_out)
        self.proj = nn.Linear(dim_out, dim_out)

    def __call__(self, x: mx.array) -> mx.array:
        """Input: [B, N, C]. Output: [B, N', dim_out] (N' = N/q_stride if q_stride>1)."""
        batch_size, num_tokens, _ = x.shape
        num_windows = (
            (num_tokens // (self.q_stride * self.window_size)) if self.use_mask_unit_attn else 1
        )

        # QKV projection + reshape to [B, N/num_windows, num_windows, 3, heads, head_dim]
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, -1, num_windows, 3, self.heads, self.head_dim)
        # Permute to [3, B, heads, num_windows, tokens_per_window, head_dim]
        qkv = mx.transpose(qkv, axes=(3, 0, 4, 2, 1, 5))
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.q_stride > 1:
            # Max-pool over q_stride tokens in the query
            # [B, heads, num_windows, q_stride, tokens/q_stride, head_dim]
            q = q.reshape(batch_size, self.heads, num_windows, self.q_stride, -1, self.head_dim)
            q = mx.max(q, axis=3)

        # Scaled dot-product attention (manual implementation)
        attn = (q * self.scale) @ mx.transpose(k, axes=(0, 1, 2, 4, 3))
        attn = mx.softmax(attn, axis=-1)
        x = attn @ v

        # [B, heads, num_windows, tokens, head_dim] -> [B, tokens, num_windows, heads, head_dim]
        # -> [B, N', dim_out]  (matches PyTorch transpose(1, 3))
        x = mx.transpose(x, axes=(0, 3, 2, 1, 4))
        x = x.reshape(batch_size, -1, self.dim_out)

        return self.proj(x)


class HieraBlock(nn.Module):
    """Single Hiera transformer block.

    At transition blocks (dim != dim_out): proj Linear + q max-pool reduces tokens 4x.
    No DropPath or LayerScale at inference (drop_path=0, init_values=None).
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.do_expand = dim != dim_out
        self.q_stride = q_stride

        self.norm1 = nn.LayerNorm(dim)
        if self.do_expand:
            self.proj = nn.Linear(dim, dim_out)

        self.attn = MaskUnitAttention(
            dim, dim_out, heads, q_stride, window_size, use_mask_unit_attn
        )

        self.norm2 = nn.LayerNorm(dim_out)
        self.mlp = HieraMLP(dim_out)

    def __call__(self, x: mx.array) -> mx.array:
        # Attention + optional Q pooling
        x_norm = self.norm1(x)
        if self.do_expand:
            x = self.proj(x_norm)
            # Max-pool over q_stride tokens to reduce spatial resolution
            batch_size = x.shape[0]
            x = x.reshape(batch_size, self.q_stride, -1, x.shape[-1])
            x = mx.max(x, axis=1)
        x = x + self.attn(x_norm)

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class HieraBackbone(nn.Module):
    """Full Hiera backbone: patch_embed + pos_embed + unroll + 24 blocks + reroll.

    Outputs 4 multiscale feature maps in NHWC at stage_ends [1, 4, 20, 23].
    Hardcoded for hiera_base_plus_224 config with 4-channel input.
    """

    def __init__(self, img_size: int = 512) -> None:
        super().__init__()
        self.img_size = img_size

        # Spatial size after patching
        self.tokens_spatial_shape = [img_size // PATCH_STRIDE[0], img_size // PATCH_STRIDE[1]]

        # Stage ends and q_pool blocks
        self.stage_ends = [sum(STAGES[:i]) - 1 for i in range(1, len(STAGES) + 1)]
        # [1, 4, 20, 23]
        q_pool_blocks = [self.stage_ends[i] + 1 for i in range(Q_POOL)]
        # [2, 5, 21]

        # Unroll schedule
        unroll_schedule = [Q_STRIDE] * len(self.stage_ends[:-1])
        # [(2,2), (2,2), (2,2)]

        # Precompute reroll schedule map
        self._reroll_schedule: dict[int, tuple[list[tuple[int, int]], list[int]]] = {}
        size = list(self.tokens_spatial_shape)
        cur_schedule = list(unroll_schedule)
        for i in range(self.stage_ends[-1] + 1):
            self._reroll_schedule[i] = (list(cur_schedule), list(size))
            if i in self.stage_ends[:Q_POOL]:
                if len(cur_schedule) > 0:
                    size = [n // s for n, s in zip(size, cur_schedule[0], strict=True)]
                cur_schedule = cur_schedule[1:]

        # Store unroll params
        self._unroll_spatial = list(self.tokens_spatial_shape)
        self._unroll_schedule = unroll_schedule

        # Patch embedding
        self.patch_embed = HieraPatchEmbed()

        # Positional embedding — placeholder overwritten by load_checkpoint().
        # Declared as mx.array so load_weights() can assign it; frozen via .eval().
        self.pos_embed = mx.zeros((1, _prod(self.tokens_spatial_shape), EMBED_DIM))

        # Build all 24 blocks
        self.blocks: list[HieraBlock] = []
        embed_dim = EMBED_DIM
        num_heads = NUM_HEADS
        flat_mu_size = _prod(MASK_UNIT_SIZE)
        flat_q_stride = _prod(Q_STRIDE)
        cur_stage = 0

        for i in range(sum(STAGES)):
            dim_out = embed_dim
            use_mu_attn = MASK_UNIT_ATTN[cur_stage]

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * DIM_MUL)
                num_heads = int(num_heads * HEAD_MUL)
                cur_stage += 1
                if i in q_pool_blocks:
                    flat_mu_size //= flat_q_stride

            block = HieraBlock(
                dim=embed_dim,
                dim_out=dim_out,
                heads=num_heads,
                q_stride=(flat_q_stride if i in q_pool_blocks else 1),
                window_size=flat_mu_size,
                use_mask_unit_attn=use_mu_attn,
            )
            self.blocks.append(block)
            embed_dim = dim_out

    def load_checkpoint(self, path: str | Path) -> None:
        """Load weights from converted safetensors checkpoint.

        Strips ``encoder.model.`` prefix and bicubic-interpolates pos_embed
        from training resolution to model resolution.
        """
        target_tokens = _prod(self.tokens_spatial_shape)
        weight_pairs: list[tuple[str, mx.array]] = []

        with safe_open(str(path), framework="numpy") as f:
            for full_key in f.keys():  # noqa: SIM118 — safe_open isn't iterable
                if not full_key.startswith(ENCODER_KEY_PREFIX):
                    continue
                mlx_key = full_key[len(ENCODER_KEY_PREFIX) :]
                tensor = mx.array(f.get_tensor(full_key))

                if mlx_key == "pos_embed":
                    tensor = _interpolate_pos_embed(tensor, target_tokens)
                    # materialize interpolated embedding
                    mx.eval(tensor)  # noqa: S307 — mx.eval, not Python eval

                weight_pairs.append((mlx_key, tensor))

        self.load_weights(weight_pairs)
        self.eval()
        # materialize all parameters
        mx.eval(self.parameters())  # noqa: S307 — mx.eval, not Python eval

    def __call__(self, x: mx.array) -> list[mx.array]:
        """Forward pass.

        Args:
            x: Input image (B, H, W, 4) in NHWC — ImageNet-normalized RGB + alpha hint.

        Returns:
            4 feature maps in NHWC:
                [0]: (B, H/4,  W/4,  112)
                [1]: (B, H/8,  W/8,  224)
                [2]: (B, H/16, W/16, 448)
                [3]: (B, H/32, W/32, 896)
        """
        # Patch embed -> [B, N, C]
        x = self.patch_embed(x)

        # Add positional embedding
        x = x + self.pos_embed

        # Unroll for windowed attention
        x = unroll(x, self._unroll_spatial, self._unroll_schedule)

        # Run blocks, collecting features at stage_ends
        features: list[mx.array] = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.stage_ends:
                feat = reroll(x, i, self._reroll_schedule)
                features.append(feat)

        return features
