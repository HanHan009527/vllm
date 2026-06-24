# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn

from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    dequantize_fp8_block_weight,
)


def test_dequantize_fp8_block_weight_uses_direct_scale():
    weight = torch.ones((4, 4), dtype=torch.float32)
    scale = torch.tensor([[2.0, 4.0], [8.0, 16.0]], dtype=torch.float32)

    out = dequantize_fp8_block_weight(weight, scale, [2, 2])

    expected = torch.tensor(
        [
            [2.0, 2.0, 4.0, 4.0],
            [2.0, 2.0, 4.0, 4.0],
            [8.0, 8.0, 16.0, 16.0],
            [8.0, 8.0, 16.0, 16.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(out, expected)


def test_fp8_linear_method_caches_bmm_bf16_weight_from_direct_scale():
    method = object.__new__(Fp8LinearMethod)
    method.block_quant = True
    method.weight_block_size = [2, 2]

    layer = nn.Module()
    layer.is_bmm = True
    layer.weight = nn.Parameter(torch.ones((4, 4), dtype=torch.float32))
    layer.weight_scale_inv = nn.Parameter(
        torch.tensor([[2.0, 4.0], [8.0, 16.0]], dtype=torch.float32)
    )

    method._cache_bf16_weight_if_needed(layer)

    expected = torch.tensor(
        [
            [2.0, 2.0, 4.0, 4.0],
            [2.0, 2.0, 4.0, 4.0],
            [8.0, 8.0, 16.0, 16.0],
            [8.0, 8.0, 16.0, 16.0],
        ],
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(layer._fp8_weight_bf16, expected)
    torch.testing.assert_close(layer._fp8_bmm_weight_bf16, expected)


def test_dequantize_fp8_block_weight_uses_inverse_scale():
    weight = torch.ones((4, 4), dtype=torch.float32)
    scale_inv = torch.tensor([[2.0, 4.0], [8.0, 16.0]], dtype=torch.float32)

    out = dequantize_fp8_block_weight(
        weight, scale_inv, [2, 2], scale_is_inverse=True
    )

    expected = torch.tensor(
        [
            [0.5, 0.5, 0.25, 0.25],
            [0.5, 0.5, 0.25, 0.25],
            [0.125, 0.125, 0.0625, 0.0625],
            [0.125, 0.125, 0.0625, 0.0625],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(out, expected)
