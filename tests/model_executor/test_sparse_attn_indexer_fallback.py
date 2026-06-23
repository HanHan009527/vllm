# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.sparse_attn_indexer import (
    _fp8_mqa_logits_torch_fallback,
    _fp8_paged_mqa_logits_torch_fallback,
)


def test_fp8_mqa_logits_torch_fallback_matches_reference():
    q = torch.tensor(
        [
            [[1.0, 2.0], [-1.0, 1.0]],
            [[0.5, -1.0], [2.0, 0.25]],
        ],
        dtype=torch.float32,
    )
    kv_values = torch.tensor(
        [[1.0, 0.5], [-1.0, 2.0], [0.25, -0.5]], dtype=torch.float32
    )
    kv_scales = torch.tensor([1.0, 0.5, 2.0], dtype=torch.float32)
    weights = torch.tensor([[0.5, 1.25], [1.5, 0.75]], dtype=torch.float32)
    cu_seqlen_ks = torch.tensor([0, 1], dtype=torch.int32)
    cu_seqlen_ke = torch.tensor([2, 3], dtype=torch.int32)

    actual = _fp8_mqa_logits_torch_fallback(
        (q, None), (kv_values, kv_scales), weights, cu_seqlen_ks, cu_seqlen_ke
    )

    kv = kv_values * kv_scales[:, None]
    score = torch.einsum("mhd,nd->hmn", q, kv)
    expected = (score.relu() * weights.transpose(0, 1).unsqueeze(-1)).sum(dim=0)
    positions = torch.arange(kv_values.shape[0])
    mask = (positions[None, :] >= cu_seqlen_ks[:, None]) & (
        positions[None, :] < cu_seqlen_ke[:, None]
    )
    expected = expected.masked_fill(~mask, float("-inf"))

    torch.testing.assert_close(actual, expected)


def test_fp8_mqa_logits_torch_fallback_rejects_fp4_q():
    q_values = torch.zeros((1, 1, 2), dtype=torch.uint8)
    q_scales = torch.zeros((1, 1), dtype=torch.uint8)
    kv_values = torch.zeros((1, 2), dtype=torch.float32)
    kv_scales = torch.ones((1,), dtype=torch.float32)
    weights = torch.ones((1, 1), dtype=torch.float32)
    cu_seqlen = torch.zeros((1,), dtype=torch.int32)

    with pytest.raises(RuntimeError, match="only supports the FP8"):
        _fp8_mqa_logits_torch_fallback(
            (q_values, q_scales),
            (kv_values, kv_scales),
            weights,
            cu_seqlen,
            cu_seqlen + 1,
        )


def test_fp8_paged_mqa_logits_torch_fallback_matches_reference():
    q = torch.tensor(
        [[[[1.0, 0.5]], [[-0.5, 2.0]]]],
        dtype=torch.float32,
    )
    kv = torch.tensor(
        [[[[1.0, 0.0]], [[0.5, 1.0]]], [[[2.0, -1.0]], [[1.0, 1.0]]]],
        dtype=torch.float32,
    )
    weights = torch.tensor([[1.0], [0.25]], dtype=torch.float32)
    context_lens = torch.tensor([[3, 4]], dtype=torch.int32)
    block_tables = torch.tensor([[0, 1]], dtype=torch.int32)
    max_model_len = 4

    kv_cache = torch.empty((2, 2, 1, 6), dtype=torch.uint8)
    kv_cache[..., :2] = kv.to(torch.float8_e4m3fn).view(torch.uint8)
    scale = torch.ones((2, 2, 1, 1), dtype=torch.float32)
    kv_cache[..., 2:6] = scale.view(torch.uint8)

    actual = _fp8_paged_mqa_logits_torch_fallback(
        (q, None),
        kv_cache,
        weights,
        context_lens,
        block_tables,
        max_model_len=max_model_len,
    )

    expected = torch.full((2, max_model_len), float("-inf"), dtype=torch.float32)
    flat_k = kv.reshape(4, 2)
    q_offsets = context_lens[0] - 1
    for row in range(2):
        for col in range(int(q_offsets[row].item()) + 1):
            score = (q[0, row, 0] * flat_k[col]).sum()
            expected[row, col] = score.relu() * weights[row, 0]

    torch.testing.assert_close(actual, expected)
