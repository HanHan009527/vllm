# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.models.deepseek_v4.nvidia.ops import o_proj


def test_deepseek_v4_fp8_einsum_groups_flat_weight_and_scale(monkeypatch):
    calls = []

    def fake_fp8_einsum(equation, lhs, rhs, out, *, recipe):
        calls.append((equation, lhs, rhs, out, recipe))

    monkeypatch.setattr(o_proj, "fp8_einsum", fake_fp8_einsum)

    a = torch.arange(2 * 2 * 256, dtype=torch.float32).reshape(2, 2, 256)
    a_scale = torch.ones(2, 2, 2)
    b = torch.arange(8 * 256, dtype=torch.float32).reshape(8, 256)
    b_scale = torch.arange(2 * 2, dtype=torch.float32).reshape(2, 2)
    out = torch.empty(2, 2, 4)

    o_proj._deepseek_v4_fp8_einsum(a, a_scale, b, b_scale, out, (1, 128, 128))

    assert len(calls) == 1
    equation, lhs, rhs, actual_out, recipe = calls[0]
    assert equation == "bhr,hdr->bhd"
    assert lhs == (a, a_scale)
    assert actual_out is out
    assert recipe == (1, 128, 128)

    grouped_b, grouped_b_scale = rhs
    assert grouped_b.shape == (2, 4, 256)
    assert grouped_b_scale.shape == (2, 1, 2)
    torch.testing.assert_close(grouped_b, b.view(2, 4, 256))
    torch.testing.assert_close(grouped_b_scale, b_scale.view(2, 1, 2))


def test_deepseek_v4_fp8_einsum_narrows_weight_group_partition(monkeypatch):
    calls = []

    def fake_fp8_einsum(equation, lhs, rhs, out, *, recipe):
        calls.append((equation, lhs, rhs, out, recipe))

    monkeypatch.setattr(o_proj, "fp8_einsum", fake_fp8_einsum)
    monkeypatch.setattr(o_proj, "get_tensor_model_parallel_rank", lambda: 1)

    a = torch.zeros(1, 2, 256)
    a_scale = torch.ones(1, 2, 2)
    b = torch.arange(16 * 256, dtype=torch.float32).reshape(16, 256)
    b_scale = torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2)
    out = torch.empty(1, 2, 4)

    o_proj._deepseek_v4_fp8_einsum(a, a_scale, b, b_scale, out, (1, 128, 128))

    grouped_b, grouped_b_scale = calls[0][2]
    assert grouped_b.shape == (2, 4, 256)
    assert grouped_b_scale.shape == (2, 1, 2)
    torch.testing.assert_close(grouped_b, b.view(4, 4, 256).narrow(0, 2, 2))
    torch.testing.assert_close(grouped_b_scale, b_scale.view(4, 1, 2).narrow(0, 2, 2))


def test_deepseek_v4_fp8_einsum_decodes_e8m0_scales(monkeypatch):
    calls = []

    def fake_fp8_einsum(equation, lhs, rhs, out, *, recipe):
        calls.append((equation, lhs, rhs, out, recipe))

    monkeypatch.setattr(o_proj, "fp8_einsum", fake_fp8_einsum)

    a = torch.zeros(1, 2, 256)
    a_scale = torch.tensor([127, 128, 129, 130], dtype=torch.uint8).reshape(1, 2, 2)
    b = torch.zeros(8, 256)
    b_scale = torch.tensor([127, 128, 129, 130], dtype=torch.uint8).reshape(2, 2)
    out = torch.empty(1, 2, 4)

    o_proj._deepseek_v4_fp8_einsum(a, a_scale, b, b_scale, out, (1, 128, 128))

    decoded_a_scale = calls[0][1][1]
    decoded_b_scale = calls[0][2][1]
    assert decoded_a_scale.dtype == torch.float32
    assert decoded_b_scale.dtype == torch.float32
    assert decoded_a_scale.is_contiguous()
    assert decoded_b_scale.is_contiguous()
    torch.testing.assert_close(
        decoded_a_scale, torch.tensor([1.0, 2.0, 4.0, 8.0]).reshape(1, 2, 2)
    )
    torch.testing.assert_close(
        decoded_b_scale, torch.tensor([1.0, 2.0, 4.0, 8.0]).reshape(2, 1, 2)
    )
