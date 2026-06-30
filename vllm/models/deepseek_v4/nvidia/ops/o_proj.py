# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.distributed import get_tensor_model_parallel_rank
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum


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


def _deepseek_v4_fp8_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    recipe: tuple[int, int, int],
) -> None:
    """Run DeepSeek V4 grouped wo_a einsum with explicit grouped metadata.

    DeepGEMM's ``bhr,hdr->bhd`` path expects the rhs weight and block scales to
    carry the output group dimension.  ``wo_a`` stores them as flat linear
    parameters, so reshape/narrow them before crossing into DeepGEMM.
    """
    if b.dim() == 2:
        num_groups = out.shape[1]
        out_rank = out.shape[2]
        hidden_size = a.shape[2]
        if b.shape[0] % out_rank != 0:
            raise RuntimeError(
                "DeepSeek V4 fp8 einsum weight rows must be divisible by "
                f"out_rank={out_rank}, got {b.shape[0]}"
            )

        b_groups = b.shape[0] // out_rank
        group_start = 0
        if b_groups != num_groups:
            if b_groups % num_groups != 0:
                raise RuntimeError(
                    "DeepSeek V4 fp8 einsum weight groups must match the "
                    "TP-local output groups or be an integer multiple of "
                    f"them, got weight_groups={b_groups}, "
                    f"output_groups={num_groups}"
                )
            group_partitions = b_groups // num_groups
            group_start = (
                get_tensor_model_parallel_rank() % group_partitions
            ) * num_groups

        b = b.view(b_groups, out_rank, hidden_size)
        if group_start != 0 or b_groups != num_groups:
            b = b.narrow(0, group_start, num_groups)

        if b_scale.dim() == 2:
            scale_mn = recipe[1]
            scale_k_pack = 4 if b_scale.dtype == torch.int32 else 1
            scale_k = recipe[2] * scale_k_pack
            scale_out_blocks = (out_rank + scale_mn - 1) // scale_mn
            scale_hidden_blocks = (hidden_size + scale_k - 1) // scale_k
            if b_scale.shape[0] % scale_out_blocks != 0:
                raise RuntimeError(
                    "DeepSeek V4 fp8 einsum scale rows must be divisible by "
                    f"scale_out_blocks={scale_out_blocks}, got "
                    f"{b_scale.shape[0]}"
                )

            scale_groups = b_scale.shape[0] // scale_out_blocks
            if scale_groups not in (num_groups, b_groups):
                raise RuntimeError(
                    "DeepSeek V4 fp8 einsum scale groups must match the "
                    "TP-local output groups or weight groups, got "
                    f"scale_groups={scale_groups}, output_groups={num_groups}, "
                    f"weight_groups={b_groups}"
                )
            b_scale = b_scale.view(
                scale_groups,
                scale_out_blocks,
                scale_hidden_blocks,
            )
            if scale_groups == b_groups and scale_groups != num_groups:
                b_scale = b_scale.narrow(0, group_start, num_groups)
        elif b_scale.dim() == 3 and b_scale.shape[0] == b_groups:
            if b_groups != num_groups:
                b_scale = b_scale.narrow(0, group_start, num_groups)

    fp8_einsum("bhr,hdr->bhd", (a, a_scale), (b, b_scale), out, recipe=recipe)


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
) -> torch.Tensor:
    """O projection: inverse RoPE + FP8 quant + einsum + wo_b.

    Shared by the FlashMLA and FlashInfer CUDA backends. ``einsum_recipe`` /
    ``tma_aligned_scales`` come from ``compute_fp8_einsum_recipe``.
    """
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
    # MarlinFP8.process_weights_after_loading renames block-FP8 scales to
    # weight_scale_inv. Non-Marlin kernels keep the on-disk weight_scale name.
    wo_a_scale = getattr(wo_a, "weight_scale_inv", None)
    if wo_a_scale is None:
        wo_a_scale = wo_a.weight_scale
    _deepseek_v4_fp8_einsum(
        o_fp8,
        o_scale,
        wo_a.weight,
        wo_a_scale,
        z,
        einsum_recipe,
    )
    return wo_b(z.flatten(1))
