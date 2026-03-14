"""Custom Metal GroupNorm — eliminates NHWC↔NCHW transposes.

Two-kernel approach with multi-threadgroup parallelism:
  Kernel A (stats): many threadgroups per group, shared-mem reduction,
    ONE atomic per threadgroup for cross-group aggregation
  Kernel B (normalize): fully parallel over all elements

~67% faster than nn.GroupNorm(pytorch_compatible=True) at 1024² and 2048².
"""

from __future__ import annotations

import mlx.core as mx

# Refiner constants — all 9 GroupNorm instances use these
_NUM_GROUPS = 8
_GROUP_SIZE = 8  # 64 channels / 8 groups
_NUM_CHANNELS = 64
_MAX_SIMD_GROUPS = 32  # 1024 threads / 32 SIMD width
_CHUNKS_PER_GROUP = 16  # Optimal for 1024² and 2048²
_THREADS_PER_TG = 1024


def _build_stats_kernel() -> object:
    """Build stats reduction kernel (many threadgroups, shared-mem + 1 atomic each)."""
    source = f"""
    const uint G = {_NUM_GROUPS};
    const uint GS = {_GROUP_SIZE};
    const uint C = {_NUM_CHANNELS};

    // chunks_per_group passed via chunks_shape[0]
    uint cpg = chunks_shape[0];

    // Decode threadgroup ID → (batch, group, chunk)
    uint tg_id = threadgroup_position_in_grid.x;
    uint chunk = tg_id % cpg;
    uint bg = tg_id / cpg;
    uint g = bg % G;
    uint b = bg / G;

    uint tid = thread_position_in_threadgroup.x;
    uint num_threads = threads_per_threadgroup.x;
    const uint SIMD_WIDTH = 32;
    uint simd_lane = tid % SIMD_WIDTH;
    uint simd_gid = tid / SIMD_WIDTH;
    uint num_simd_groups = (num_threads + SIMD_WIDTH - 1) / SIMD_WIDTH;

    uint H = inp_shape[1];
    uint W = inp_shape[2];
    uint HW = H * W;
    uint group_elems = HW * GS;

    // This chunk's range within the group
    uint chunk_size = (group_elems + cpg - 1) / cpg;
    uint chunk_start = chunk * chunk_size;
    uint chunk_end = min(chunk_start + chunk_size, group_elems);

    uint batch_offset = b * HW * C;
    uint group_chan_offset = g * GS;

    // ── Accumulate partial sums for this chunk ──
    float local_sum = 0.0f;
    float local_sumsq = 0.0f;

    for (uint i = chunk_start + tid; i < chunk_end; i += num_threads) {{
        uint spatial = i / GS;
        uint c_in_group = i % GS;
        uint idx = batch_offset + spatial * C + group_chan_offset + c_in_group;
        float val = static_cast<float>(inp[idx]);
        local_sum += val;
        local_sumsq += val * val;
    }}

    // SIMD reduction
    float simd_s = simd_sum(local_sum);
    float simd_sq = simd_sum(local_sumsq);

    threadgroup float shared_sum[{_MAX_SIMD_GROUPS}];
    threadgroup float shared_sumsq[{_MAX_SIMD_GROUPS}];

    if (simd_lane == 0) {{
        shared_sum[simd_gid] = simd_s;
        shared_sumsq[simd_gid] = simd_sq;
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction within this threadgroup
    if (simd_gid == 0) {{
        float s = (simd_lane < num_simd_groups) ? shared_sum[simd_lane] : 0.0f;
        float sq = (simd_lane < num_simd_groups) ? shared_sumsq[simd_lane] : 0.0f;
        s = simd_sum(s);
        sq = simd_sum(sq);

        // ONE atomic per threadgroup (not per SIMD group)
        if (simd_lane == 0) {{
            uint stats_idx = (b * G + g) * 2;
            atomic_fetch_add_explicit(&stats[stats_idx], s, memory_order_relaxed);
            atomic_fetch_add_explicit(&stats[stats_idx + 1], sq, memory_order_relaxed);
        }}
    }}
    """
    return mx.fast.metal_kernel(
        name="groupnorm_stats_v2",
        input_names=["inp", "chunks"],
        output_names=["stats"],
        source=source,
        ensure_row_contiguous=False,
        atomic_outputs=True,
    )


def _build_norm_kernel(relu: bool = False) -> object:
    """Build normalize kernel (fully parallel over all elements)."""
    relu_code = "result = result > 0.0f ? result : 0.0f;" if relu else ""
    source = f"""
    const uint G = {_NUM_GROUPS};
    const uint GS = {_GROUP_SIZE};
    const uint C = {_NUM_CHANNELS};

    uint elem = thread_position_in_grid.x;
    uint H = inp_shape[1];
    uint W = inp_shape[2];
    uint HW = H * W;
    uint total = inp_shape[0] * HW * C;
    if (elem >= total) return;

    uint c = elem % C;
    uint spatial = (elem / C) % HW;
    uint b = elem / (HW * C);
    uint g = c / GS;

    uint stats_idx = (b * G + g) * 2;
    float s = stats[stats_idx];
    float sq = stats[stats_idx + 1];
    float n = static_cast<float>(HW * GS);
    float mean = s / n;
    float var = sq / n - mean * mean;
    float inv_std = rsqrt(var + 1e-5f);

    float val = static_cast<float>(inp[elem]);
    float normed = (val - mean) * inv_std;
    float result = normed * weight[c] + bias_arr[c];
    {relu_code}
    out[elem] = static_cast<T>(result);
    """
    suffix = "_relu" if relu else ""
    return mx.fast.metal_kernel(
        name=f"groupnorm_norm_v2{suffix}",
        input_names=["inp", "stats", "weight", "bias_arr"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=False,
    )


# Lazy-initialized kernels (built on first use)
_stats_kernel: object | None = None
_norm_kernel: object | None = None
_chunks_dummy: mx.array | None = None


def _ensure_kernels() -> None:
    """Build Metal kernels on first use."""
    global _stats_kernel, _norm_kernel, _chunks_dummy  # noqa: PLW0603
    if _stats_kernel is None:
        _stats_kernel = _build_stats_kernel()
        _norm_kernel = _build_norm_kernel(relu=False)
        _chunks_dummy = mx.zeros((_CHUNKS_PER_GROUP,))


def metal_groupnorm(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    frozen_stats: tuple[mx.array, mx.array] | None = None,
) -> mx.array:
    """Custom Metal GroupNorm for refiner (G=8, C=64).

    Eliminates the two NHWC↔NCHW transposes that nn.GroupNorm requires.
    ~67% faster than nn.GroupNorm(pytorch_compatible=True) at 1024²+.

    Args:
        x: (B, H, W, 64) input tensor in NHWC
        weight: (64,) affine scale
        bias: (64,) affine offset
        frozen_stats: Optional (mean, var) from full-image stats collection.
            When provided, skips kernel A (stats computation) and uses these
            instead. Both arrays should be (B, G, 1) float32.

    Returns:
        Normalized tensor, same shape and dtype as x.
    """
    _ensure_kernels()
    B, H, W, C = x.shape

    # Metal kernels operate in fp32 for numerical stability — upcast if needed
    weight_f32 = weight.astype(mx.float32) if weight.dtype != mx.float32 else weight
    bias_f32 = bias.astype(mx.float32) if bias.dtype != mx.float32 else bias

    if frozen_stats is not None:
        # Convert (mean, var) to (sum, sumsq) format for kernel B
        mean, var = frozen_stats
        n = float(H * W * _GROUP_SIZE)
        # Flatten (B, G, 1) → (B*G,) then interleave as (sum0, sumsq0, sum1, sumsq1, ...)
        mean_flat = mean.reshape(B * _NUM_GROUPS)
        var_flat = var.reshape(B * _NUM_GROUPS)
        sums = mean_flat * n
        sumsqs = mx.maximum(var_flat + mean_flat * mean_flat, mx.array(0.0)) * n
        # Interleave: stats[i*2] = sum, stats[i*2+1] = sumsq
        stats = mx.stack([sums, sumsqs], axis=-1).reshape(B * _NUM_GROUPS * 2)
    else:
        # Kernel A: stats reduction
        num_tg_stats = B * _NUM_GROUPS * _CHUNKS_PER_GROUP
        stats_shape = (B * _NUM_GROUPS * 2,)
        stats = _stats_kernel(
            inputs=[x, _chunks_dummy],
            template=[("T", x.dtype)],
            output_shapes=[stats_shape],
            output_dtypes=[mx.float32],
            grid=(num_tg_stats * _THREADS_PER_TG, 1, 1),
            threadgroup=(_THREADS_PER_TG, 1, 1),
            init_value=0,
        )[0]

    # Kernel B: normalize + affine
    total_elements = B * H * W * C
    grid_size = (
        (total_elements + _THREADS_PER_TG - 1) // _THREADS_PER_TG
    ) * _THREADS_PER_TG
    out = _norm_kernel(
        inputs=[x, stats, weight_f32, bias_f32],
        template=[("T", x.dtype)],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=(grid_size, 1, 1),
        threadgroup=(_THREADS_PER_TG, 1, 1),
    )[0]

    return out
