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

        # Pre-transposed weight caches — populated by prepare_transposed_weights()
        self._fc1_wt: mx.array | None = None
        self._fc2_wt: mx.array | None = None

    def prepare_transposed_weights(self) -> None:
        """Pre-transpose fc1/fc2 weights to [in, out] layout.

        nn.Linear stores weight as [out, in] and computes x @ W.T, creating
        a non-contiguous transposed view each call. Pre-transposing materializes
        a contiguous [in, out] array so the matmul reads contiguous memory.
        Called after weights are loaded.
        """
        self._fc1_wt = mx.contiguous(self.fc1.weight.T)
        self._fc2_wt = mx.contiguous(self.fc2.weight.T)
        # materialize pre-transposed weights — mx.eval is MLX array materialization
        mx.eval(self._fc1_wt, self._fc2_wt)  # noqa: S307  -- mx.eval, not Python eval

    def __call__(self, x: mx.array) -> mx.array:
        if self._fc1_wt is not None:
            x = x @ self._fc1_wt + self.fc1.bias
            x = nn.gelu(x)
            x = x @ self._fc2_wt + self.fc2.bias
            return x
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
        use_sdpa: bool = True,
    ) -> None:
        super().__init__()
        self.dim_out = dim_out
        self.heads = heads
        self.q_stride = q_stride
        self.head_dim = dim_out // heads
        self.scale = self.head_dim**-0.5
        self.window_size = window_size
        self.use_mask_unit_attn = use_mask_unit_attn
        self._use_sdpa = use_sdpa

        self.qkv = nn.Linear(dim, 3 * dim_out)
        self.proj = nn.Linear(dim_out, dim_out)

        # Pre-split QKV weight/bias caches — populated by prepare_split_qkv()
        self._q_weight: mx.array | None = None
        self._k_weight: mx.array | None = None
        self._v_weight: mx.array | None = None
        self._q_bias: mx.array | None = None
        self._k_bias: mx.array | None = None
        self._v_bias: mx.array | None = None

    def prepare_split_qkv(self) -> None:
        """Pre-split QKV weights into contiguous Q, K, V arrays.

        Eliminates non-contiguous slices from mx.split(qkv, 3, axis=-1)
        which force implicit copies on subsequent reshapes. Each pre-split
        weight is contiguous, so x @ W.T outputs are natively contiguous.
        Must be called after weights are loaded.
        """
        w = self.qkv.weight  # [3*dim_out, dim]
        d = self.dim_out
        # Slice along first dim of row-major array — each slice is contiguous
        self._q_weight = w[:d]
        self._k_weight = w[d : 2 * d]
        self._v_weight = w[2 * d :]
        if "bias" in self.qkv:
            b = self.qkv.bias  # [3*dim_out]
            self._q_bias = b[:d]
            self._k_bias = b[d : 2 * d]
            self._v_bias = b[2 * d :]
        # Materialize pre-split weights — mx.eval is MLX graph evaluation
        mx.eval(  # noqa: S307
            self._q_weight, self._k_weight, self._v_weight,
            self._q_bias, self._k_bias, self._v_bias,
        )

    def _compute_qkv(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Compute Q, K, V with contiguous outputs.

        Uses pre-split weights (3 separate addmm) if available, otherwise
        falls back to single QKV linear + split.
        """
        if self._q_weight is not None:
            # 3 separate matmuls — each output is natively contiguous
            q = mx.addmm(self._q_bias, x, self._q_weight.T)
            k = mx.addmm(self._k_bias, x, self._k_weight.T)
            v = mx.addmm(self._v_bias, x, self._v_weight.T)
            return q, k, v
        # Fallback: single matmul + non-contiguous split
        qkv = self.qkv(x)
        return mx.split(qkv, 3, axis=-1)

    def __call__(self, x: mx.array) -> mx.array:
        """Input: [B, N, C]. Output: [B, N', dim_out] (N' = N/q_stride if q_stride>1)."""
        batch_size, num_tokens, _ = x.shape
        num_windows = (
            (num_tokens // (self.q_stride * self.window_size)) if self.use_mask_unit_attn else 1
        )

        q, k, v = self._compute_qkv(x)

        if num_windows == 1 and self._use_sdpa:
            # Fast path for global attention (stages 2-3, 19 of 24 blocks).
            # Each q/k/v is contiguous [B, N, dim_out] — reshapes are zero-copy.
            q = mx.transpose(q.reshape(batch_size, num_tokens, self.heads, self.head_dim), axes=(0, 2, 1, 3))
            k = mx.transpose(k.reshape(batch_size, num_tokens, self.heads, self.head_dim), axes=(0, 2, 1, 3))
            v = mx.transpose(v.reshape(batch_size, num_tokens, self.heads, self.head_dim), axes=(0, 2, 1, 3))

            if self.q_stride > 1:
                q = q.reshape(batch_size, self.heads, self.q_stride, -1, self.head_dim)
                q = mx.max(q, axis=2)

            x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)

            # Fused output projection: contract over heads+head_dim directly
            # on SDPA layout [B, heads, N', head_dim], avoiding the
            # transpose(0,2,1,3)+reshape copy that makes a contiguous copy
            # before the proj matmul. 19 of 24 blocks use this path.
            proj_w = self.proj.weight.reshape(self.dim_out, self.heads, self.head_dim)
            x = mx.einsum("bhnd,ohd->bno", x, proj_w)
            if self.proj.bias is not None:
                x = x + self.proj.bias
            return x
        else:
            # Windowed path for mask-unit attention (stages 0-1, 5 blocks).
            # q/k/v already computed by _compute_qkv() above — each is
            # contiguous [B, N, dim_out], so reshapes below are zero-copy.
            tokens_per_window = num_tokens // num_windows
            # [B, tokens_per_win, num_windows, heads, head_dim]
            q = q.reshape(batch_size, tokens_per_window, num_windows, self.heads, self.head_dim)
            k = k.reshape(batch_size, tokens_per_window, num_windows, self.heads, self.head_dim)
            v = v.reshape(batch_size, tokens_per_window, num_windows, self.heads, self.head_dim)

            if self.q_stride > 1:
                # Group adjacent tokens by q_stride and take max for pooling
                q = q.reshape(
                    batch_size, self.q_stride, -1, num_windows, self.heads, self.head_dim
                )
                q = mx.max(q, axis=1)

            if self._use_sdpa:
                # Transpose to [B, num_windows, heads, tokens, head_dim]
                q = mx.transpose(q, axes=(0, 2, 3, 1, 4))
                k = mx.transpose(k, axes=(0, 2, 3, 1, 4))
                v = mx.transpose(v, axes=(0, 2, 3, 1, 4))

                q_tokens = q.shape[3]
                kv_tokens = k.shape[3]
                # Flatten to 4D: [B*num_windows, heads, tokens, head_dim]
                q_4d = q.reshape(batch_size * num_windows, self.heads, q_tokens, self.head_dim)
                k_4d = k.reshape(batch_size * num_windows, self.heads, kv_tokens, self.head_dim)
                v_4d = v.reshape(batch_size * num_windows, self.heads, kv_tokens, self.head_dim)

                x = mx.fast.scaled_dot_product_attention(q_4d, k_4d, v_4d, scale=self.scale)

                # [B*W, heads, tokens, head_dim] -> [B, tokens, W, heads, head_dim]
                # Single composed transpose instead of two sequential ones
                x = x.reshape(batch_size, num_windows, self.heads, q_tokens, self.head_dim)
                x = mx.transpose(x, axes=(0, 3, 1, 2, 4))
            else:
                # Non-SDPA fallback: [B, heads, windows, tokens, head_dim]
                q = mx.transpose(q, axes=(0, 3, 2, 1, 4))
                k = mx.transpose(k, axes=(0, 3, 2, 1, 4))
                v = mx.transpose(v, axes=(0, 3, 2, 1, 4))

                attn = (q * self.scale) @ mx.transpose(k, axes=(0, 1, 2, 4, 3))
                attn = mx.softmax(attn, axis=-1)
                x = attn @ v
                # [B, heads, win, tokens, head_dim] -> [B, tokens, win, heads, head_dim]
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
        use_sdpa: bool = True,
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
            dim, dim_out, heads, q_stride, window_size, use_mask_unit_attn, use_sdpa=use_sdpa
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

    def __init__(self, img_size: int = 512, use_sdpa: bool = True) -> None:
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

        # Precompute unroll permutation — replaces 3 reshape-transpose-reshape
        # cycles (each forcing a contiguous copy) with a single indexed gather.
        n_full = _prod(self.tokens_spatial_shape)
        dummy = mx.arange(n_full, dtype=mx.float32).reshape(1, n_full, 1)
        unrolled = unroll(dummy, self._unroll_spatial, unroll_schedule)
        self._unroll_perm = unrolled.reshape(n_full).astype(mx.int32)

        # Precompute reroll permutations for each stage end — same idea.
        self._reroll_perms: dict[int, tuple[mx.array, list[int]]] = {}
        for se_idx in self.stage_ends:
            remaining_sched, se_size = self._reroll_schedule[se_idx]
            n_tokens = _prod(se_size)
            if n_tokens == 0:
                continue
            dummy_r = mx.arange(n_tokens, dtype=mx.float32).reshape(1, n_tokens, 1)
            spatial_out = reroll(dummy_r, se_idx, self._reroll_schedule)
            self._reroll_perms[se_idx] = (
                spatial_out.reshape(n_tokens).astype(mx.int32),
                list(se_size),
            )

        # Materialize all permutation indices (mx.eval is MLX graph evaluation, not Python eval)
        mx.eval(self._unroll_perm)
        for perm, _ in self._reroll_perms.values():
            mx.eval(perm)

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
                use_sdpa=use_sdpa,
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

        # Pre-split QKV weights for contiguous Q/K/V outputs
        # Pre-transpose MLP weights for contiguous matmul operands
        for blk in self.blocks:
            blk.attn.prepare_split_qkv()
            blk.mlp.prepare_transposed_weights()

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

        # Unroll via precomputed gather (single copy vs 3 reshape-transpose-reshape chains)
        x = x[:, self._unroll_perm, :]

        # Run blocks, collecting features at stage_ends
        features: list[mx.array] = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.stage_ends:
                # Reroll via precomputed gather (single copy vs multi-step chains)
                perm, size = self._reroll_perms[i]
                feat = x[:, perm, :].reshape(x.shape[0], size[0], size[1], x.shape[-1])
                features.append(feat)

        return features
