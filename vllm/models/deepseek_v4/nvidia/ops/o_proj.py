# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
import math
import os

import torch
import torch.nn as nn

from vllm.models.deepseek_v4.common.ops import fused_inv_rope_fp8_quant
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum, is_deep_gemm_fp8_einsum_supported

logger = logging.getLogger(__name__)


def dsv4_nonfinite_diag_enabled() -> bool:
    return os.environ.get("VLLM_DSV4_NONFINITE_DIAG") == "1"


def tensor_diag_summary(tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return "none"
    detached = tensor.detach()
    is_finite = torch.isfinite(detached)
    safe_abs = torch.nan_to_num(
        detached.float().abs(), nan=0.0, posinf=0.0, neginf=0.0
    )
    return (
        f"shape={tuple(detached.shape)} dtype={detached.dtype} "
        f"bad_count={(~is_finite).sum().item()} "
        f"finite={bool(is_finite.all().item())} "
        f"safe_amax={safe_abs.max().item() if safe_abs.numel() else 0.0}"
    )


def log_nonfinite_tensor(label: str, tensor: torch.Tensor, **extra: str) -> None:
    if not dsv4_nonfinite_diag_enabled():
        return
    if torch.isfinite(tensor).all():
        return
    extras = " ".join(f"{key}={value}" for key, value in extra.items())
    logger.error(
        "DeepSeek V4 o_proj BF16 fallback non-finite %s: %s %s",
        label,
        tensor_diag_summary(tensor),
        extras,
    )


def get_fp8_weight_scale(layer: nn.Module) -> torch.Tensor | None:
    if hasattr(layer, "weight_scale_inv"):
        return layer.weight_scale_inv
    return None


def maybe_unpack_linear_output(
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None],
) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def apply_bf16_linear(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    weight = getattr(layer, "_fp8_weight_bf16", None)
    if weight is None:
        out = maybe_unpack_linear_output(layer(x))
        log_nonfinite_tensor(
            "wo_b_forward_output",
            out,
            input=tensor_diag_summary(x),
            cached_weight="missing",
        )
        return out
    out = torch.nn.functional.linear(x, weight.to(dtype=x.dtype))
    log_nonfinite_tensor(
        "wo_b_bf16_linear_output",
        out,
        input=tensor_diag_summary(x),
        cached_weight=tensor_diag_summary(weight),
    )
    return out


def _decode_scale_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if scale.dtype == torch.uint8 or (
        e8m0_dtype is not None and scale.dtype == e8m0_dtype
    ):
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )

        return _upcast_e8m0_to_fp32(scale)
    return scale.to(torch.float32)


def _expand_block_scales(
    scale: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    scale = _decode_scale_to_fp32(scale)
    row_blocks, col_blocks = scale.shape[-2:]
    row_block = math.ceil(rows / row_blocks)
    col_block = math.ceil(cols / col_blocks)
    scale = torch.repeat_interleave(scale, row_block, dim=-2)[..., :rows, :]
    scale = torch.repeat_interleave(scale, col_block, dim=-1)[..., :, :cols]
    return scale


def _reshape_wo_a_weight(
    weight: torch.Tensor,
    *,
    n_groups: int,
    o_lora_rank: int,
    input_size: int,
) -> torch.Tensor | None:
    expected_numel = n_groups * o_lora_rank * input_size
    if weight.numel() != expected_numel or weight.ndim not in (2, 3):
        return None
    return weight.reshape(n_groups, o_lora_rank, input_size)


def get_wo_a_bf16_weight(
    wo_a: nn.Module,
    *,
    n_groups: int,
    o_lora_rank: int,
    input_size: int,
) -> torch.Tensor | None:
    for attr in ("_fp8_bmm_weight_bf16", "_fp8_weight_bf16", "_dsv4_wo_a_bf16"):
        weight = getattr(wo_a, attr, None)
        if weight is None:
            continue
        grouped_weight = _reshape_wo_a_weight(
            weight,
            n_groups=n_groups,
            o_lora_rank=o_lora_rank,
            input_size=input_size,
        )
        if grouped_weight is not None:
            return grouped_weight

    raw_weight = getattr(wo_a, "weight", None)
    if raw_weight is None:
        return None
    grouped_weight = _reshape_wo_a_weight(
        raw_weight,
        n_groups=n_groups,
        o_lora_rank=o_lora_rank,
        input_size=input_size,
    )
    if grouped_weight is None:
        return None

    weight_scale_inv = getattr(wo_a, "weight_scale_inv", None)
    if weight_scale_inv is not None:
        scale = _expand_block_scales(
            weight_scale_inv.reshape(n_groups, -1, weight_scale_inv.shape[-1]),
            o_lora_rank,
            input_size,
        )
        grouped_weight = grouped_weight.to(torch.float32) / scale

    grouped_weight = grouped_weight.to(torch.bfloat16)
    wo_a._dsv4_wo_a_bf16 = grouped_weight
    return grouped_weight


def inv_rope_bf16_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
) -> torch.Tensor:
    num_tokens, num_heads, head_dim = o.shape
    expected_heads = n_groups * heads_per_group
    expected_head_dim = nope_dim + rope_dim
    if num_heads != expected_heads:
        raise ValueError(f"Expected {expected_heads} heads, got {num_heads}.")
    if head_dim != expected_head_dim:
        raise ValueError(
            f"Expected head dimension {expected_head_dim}, got {head_dim}."
        )
    if rope_dim % 2 != 0:
        raise ValueError(f"rope_dim must be even, got {rope_dim}.")

    grouped = o.reshape(num_tokens, n_groups, heads_per_group, head_dim)
    projected = grouped.clone()

    rope = projected[..., nope_dim:]
    rope_pairs = rope.reshape(*rope.shape[:-1], rope_dim // 2, 2)
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos[:, None, None, :, None].to(dtype=rope.dtype)
    sin = sin[:, None, None, :, None].to(dtype=rope.dtype)

    x0 = rope_pairs[..., 0:1]
    x1 = rope_pairs[..., 1:2]
    rope_pairs.copy_(torch.cat((x0 * cos + x1 * sin, x1 * cos - x0 * sin), dim=-1))

    wo_a_input_size = heads_per_group * head_dim
    wo_a_groups = n_groups
    wo_a_input = projected.reshape(num_tokens, wo_a_groups, wo_a_input_size)

    wo_a_weight = get_wo_a_bf16_weight(
        wo_a,
        n_groups=wo_a_groups,
        o_lora_rank=o_lora_rank,
        input_size=wo_a_input_size,
    )
    if wo_a_weight is not None:
        grouped_weight = wo_a_weight.to(dtype=wo_a_input.dtype)
        out = torch.einsum("bgi,gri->bgr", wo_a_input, grouped_weight)
        log_nonfinite_tensor(
            "wo_a_bf16_einsum_output",
            out,
            input=tensor_diag_summary(wo_a_input),
            cached_weight=tensor_diag_summary(wo_a_weight),
        )
        return out

    if getattr(wo_a, "is_bmm", False):
        raise RuntimeError(
            "DeepSeek V4 O-proj BF16 fallback requires a grouped BF16 wo_a "
            "weight when wo_a.is_bmm is set."
        )

    out = maybe_unpack_linear_output(wo_a(wo_a_input))
    log_nonfinite_tensor(
        "wo_a_forward_output",
        out,
        input=tensor_diag_summary(wo_a_input),
        cached_weight="missing_or_unusable",
    )
    return out


def compute_fp8_einsum_recipe() -> tuple[tuple[int, int, int], bool]:
    """fp8_einsum recipe + scale layout for the current GPU arch.

    SM90: FP32 block scales stay [g, r/128, d/128] → sfb_gran_mn=128.
    SM100: INT32 packed scales become [g, r, ...] → sfb_gran_mn=1.

    Returns ``(einsum_recipe, tma_aligned_scales)`` for ``deep_gemm_fp8_o_proj``.
    """
    cap = current_platform.get_device_capability()
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)
    tma_aligned_scales = cap.major >= 10
    return einsum_recipe, tma_aligned_scales


def should_fallback_fp8_einsum_error(exc: RuntimeError) -> bool:
    message = str(exc)
    return (
        "DeepGEMM backend is not available" in message
        or "fp8_einsum" in message
        or "deepgemm" in message.lower()
        or "layout.hpp" in message
    )


def bf16_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
) -> torch.Tensor:
    z = inv_rope_bf16_o_proj(
        o,
        positions,
        cos_sin_cache,
        wo_a,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        o_lora_rank=o_lora_rank,
    )
    log_nonfinite_tensor(
        "wo_a_output_z",
        z,
        wo_a_cache=tensor_diag_summary(getattr(wo_a, "_fp8_bmm_weight_bf16", None)),
        wo_b_cache=tensor_diag_summary(getattr(wo_b, "_fp8_weight_bf16", None)),
    )
    return apply_bf16_linear(wo_b, z.flatten(1))


def deep_gemm_fp8_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
    force_bf16: bool = False,
) -> torch.Tensor:
    """O projection: inverse RoPE + FP8 quant + einsum + wo_b.

    Shared by the FlashMLA and FlashInfer CUDA backends. ``einsum_recipe`` /
    ``tma_aligned_scales`` come from ``compute_fp8_einsum_recipe``.
    """
    weight_scale = get_fp8_weight_scale(wo_a)
    if (
        force_bf16
        or weight_scale is None
        or not is_deep_gemm_fp8_einsum_supported()
    ):
        return bf16_o_proj(
            o,
            positions,
            cos_sin_cache,
            wo_a,
            wo_b,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            o_lora_rank=o_lora_rank,
        )

    o_fp8, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        tma_aligned_scales=tma_aligned_scales,
    )
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    try:
        fp8_einsum(
            "bhr,hdr->bhd",
            (o_fp8, o_scale),
            (wo_a.weight, weight_scale),
            z,
            recipe=einsum_recipe,
        )
    except RuntimeError as exc:
        if not should_fallback_fp8_einsum_error(exc):
            raise
        logger.warning(
            "DeepSeek V4 O-proj FP8 einsum failed; falling back to BF16: %s",
            exc,
        )
        return bf16_o_proj(
            o,
            positions,
            cos_sin_cache,
            wo_a,
            wo_b,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            o_lora_rank=o_lora_rank,
        )
    return wo_b(z.flatten(1))
