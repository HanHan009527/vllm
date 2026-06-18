# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4MegaMoEExperts,
    make_deepseek_v4_expert_params_mapping,
)


def _make_vllm_config():
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )


def _make_loader_only_experts(
    *,
    expert_dtype: str,
    hidden_size: int,
    intermediate_size: int,
):
    experts = DeepseekV4MegaMoEExperts.__new__(DeepseekV4MegaMoEExperts)
    experts.expert_dtype = expert_dtype
    experts.hidden_size = hidden_size
    experts.intermediate_size = intermediate_size
    experts.experts_start_idx = 0
    experts.experts_end_idx = 1
    experts.num_logical_experts = 1
    return experts


def _ceil_div_128(value: int) -> int:
    return (value + 127) // 128


def test_deepseek_v4_mega_moe_expert_mapping():
    mapping = make_deepseek_v4_expert_params_mapping(2)

    assert mapping == [
        ("experts.w13_", "experts.0.w1.", 0, "w1"),
        ("experts.w2_", "experts.0.w2.", 0, "w2"),
        ("experts.w13_", "experts.0.w3.", 0, "w3"),
        ("experts.w13_", "experts.1.w1.", 1, "w1"),
        ("experts.w2_", "experts.1.w2.", 1, "w2"),
        ("experts.w13_", "experts.1.w3.", 1, "w3"),
    ]


def test_deepseek_v4_mega_moe_ue8m0_uint8_to_float():
    raw = torch.tensor([0, 126, 127, 128], dtype=torch.uint8)

    decoded = DeepseekV4MegaMoEExperts._ue8m0_uint8_to_float(raw)

    assert torch.equal(decoded.view(torch.int32), raw.to(torch.int32) << 23)
    assert decoded[0].item() == 0.0
    assert decoded[1].item() == 0.5
    assert decoded[2].item() == 1.0
    assert decoded[3].item() == 2.0


def test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_ownership():
    experts = DeepseekV4MegaMoEExperts(
        _make_vllm_config(),
        num_experts=4,
        num_local_experts=2,
        experts_start_idx=2,
        top_k=2,
        hidden_size=128,
        intermediate_size=128,
    )

    nonlocal_weight = torch.ones(128, 64, dtype=torch.uint8)
    assert (
        experts.weight_loader(
            experts.w13_weight,
            nonlocal_weight,
            "experts.w13_weight",
            shard_id="w1",
            expert_id=1,
            return_success=True,
        )
        is False
    )

    w1 = torch.full((128, 64), 3, dtype=torch.uint8)
    w3 = torch.full((128, 64), 7, dtype=torch.uint8)
    w2 = torch.full((128, 64), 11, dtype=torch.uint8)

    assert experts.weight_loader(
        experts.w13_weight,
        w1,
        "experts.w13_weight",
        shard_id="w1",
        expert_id=2,
        return_success=True,
    )
    assert experts.weight_loader(
        experts.w13_weight,
        w3,
        "experts.w13_weight",
        shard_id="w3",
        expert_id=2,
        return_success=True,
    )
    assert experts.weight_loader(
        experts.w2_weight,
        w2,
        "experts.w2_weight",
        shard_id="w2",
        expert_id=2,
        return_success=True,
    )

    assert torch.equal(experts.w13_weight[0, :128], w1)
    assert torch.equal(experts.w13_weight[0, 128:], w3)
    assert torch.equal(experts.w2_weight[0], w2)
    assert torch.count_nonzero(experts.w13_weight[1]) == 0


def test_deepseek_v4_mega_moe_weight_loader_expands_fp8_compact_w13_scales():
    hidden_size = 16_384
    intermediate_size = 262_144
    experts = _make_loader_only_experts(
        expert_dtype="fp8",
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    w13_scale = torch.nn.Parameter(
        torch.zeros(
            1,
            2 * _ceil_div_128(intermediate_size),
            _ceil_div_128(hidden_size),
        ),
        requires_grad=False,
    )

    compact = torch.arange(16 * 32, dtype=torch.float32).reshape(16, 32)

    assert experts.weight_loader(
        w13_scale,
        compact,
        "layers.0.ffn.experts.w13_weight_scale",
        shard_id="w1",
        expert_id=0,
        return_success=True,
    )

    expanded = compact.repeat_interleave(128, dim=0).repeat_interleave(4, dim=1)
    assert torch.equal(w13_scale[0, :2048], expanded)
    assert torch.count_nonzero(w13_scale[0, 2048:]) == 0


def test_deepseek_v4_mega_moe_weight_loader_rejects_unknown_fp8_scale_layout():
    hidden_size = 16_384
    intermediate_size = 262_144
    experts = _make_loader_only_experts(
        expert_dtype="fp8",
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    w13_scale = torch.nn.Parameter(
        torch.zeros(
            1,
            2 * _ceil_div_128(intermediate_size),
            _ceil_div_128(hidden_size),
        ),
        requires_grad=False,
    )

    with pytest.raises(ValueError, match="compact FP8 w1/w3 scale layout"):
        experts.weight_loader(
            w13_scale,
            torch.zeros(17, 32),
            "layers.0.ffn.experts.w13_weight_scale",
            shard_id="w1",
            expert_id=0,
            return_success=True,
        )


def test_deepseek_v4_mega_moe_weight_loader_keeps_fp4_scale_layout():
    hidden_size = 256
    intermediate_size = 256
    experts = _make_loader_only_experts(
        expert_dtype="fp4",
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    w13_scale = torch.nn.Parameter(
        torch.zeros(1, 2 * intermediate_size, hidden_size // 2),
        requires_grad=False,
    )
    w1_scale = torch.arange(
        intermediate_size * (hidden_size // 2),
        dtype=torch.float32,
    ).reshape(intermediate_size, hidden_size // 2)

    assert experts.weight_loader(
        w13_scale,
        w1_scale,
        "layers.0.ffn.experts.w13_weight_scale",
        shard_id="w1",
        expert_id=0,
        return_success=True,
    )

    assert torch.equal(w13_scale[0, :intermediate_size], w1_scale)
    assert torch.count_nonzero(w13_scale[0, intermediate_size:]) == 0


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DeepSeek V4 MegaMoE fused input staging requires CUDA.",
)
def test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact():
    from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import (
        prepare_megamoe_inputs,
    )
    from vllm.third_party.deep_gemm.utils import per_token_cast_to_fp8

    device = torch.device("cuda")
    num_tokens = 7
    hidden_size = 256
    top_k = 8

    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    hidden_states = (
        torch.randn(
            num_tokens,
            hidden_size,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * 17.0
    ).to(torch.bfloat16)
    hidden_states[0, :32] = 0
    hidden_states[1, 32:64] = 1.0e-6
    hidden_states[2, 64:96] = -1.0e-6

    topk_ids = torch.randint(
        0,
        256,
        (num_tokens, top_k),
        device=device,
        dtype=torch.int32,
        generator=generator,
    )
    topk_weights = torch.randn(
        num_tokens,
        top_k,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    ref_x, ref_x_sf = per_token_cast_to_fp8(
        hidden_states,
        use_ue8m0=True,
        gran_k=32,
        use_packed_ue8m0=True,
    )
    ref_topk_idx = topk_ids.to(torch.int64)
    ref_topk_weights = topk_weights.clone()

    fused_x = torch.empty_like(ref_x)
    fused_x_sf = torch.empty_like(ref_x_sf)
    fused_topk_idx = torch.empty_like(ref_topk_idx)
    fused_topk_weights = torch.empty_like(ref_topk_weights)

    prepare_megamoe_inputs(
        hidden_states,
        topk_weights,
        topk_ids,
        fused_x,
        fused_x_sf,
        fused_topk_idx,
        fused_topk_weights,
    )
    torch.accelerator.synchronize()

    assert torch.equal(fused_x.view(torch.uint8), ref_x.view(torch.uint8))
    assert torch.equal(fused_x_sf, ref_x_sf)
    assert torch.equal(fused_topk_idx, ref_topk_idx)
    assert torch.equal(
        fused_topk_weights.view(torch.uint8),
        ref_topk_weights.view(torch.uint8),
    )
