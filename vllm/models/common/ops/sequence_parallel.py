# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.pcp_performance import pcp_performance_trace


def _custom_collective(name: str, x: torch.Tensor) -> torch.Tensor | None:
    device_communicator = get_tp_group().device_communicator
    if device_communicator is None:
        return None
    collective = getattr(device_communicator, name, None)
    return None if collective is None else collective(x)


def sp_all_gather(x: torch.Tensor) -> torch.Tensor:
    if not pcp_performance_trace.enabled:
        output = _custom_collective("custom_all_gather", x)
        if output is not None:
            return output
        return tensor_model_parallel_all_gather(x, 0)

    tp_size = get_tensor_model_parallel_world_size()
    local_payload_bytes = x.numel() * x.element_size()
    with pcp_performance_trace.region(
        "tp.sp_all_gather",
        local_payload_bytes=local_payload_bytes,
        gathered_payload_bytes=local_payload_bytes * tp_size,
        group_size=tp_size,
    ):
        output = _custom_collective("custom_all_gather", x)
        if output is not None:
            return output
        return tensor_model_parallel_all_gather(x, 0)


def sp_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2
    tp_size = get_tensor_model_parallel_world_size()
    sp_pad = (-x.shape[0]) % tp_size
    if sp_pad > 0:
        x = torch.nn.functional.pad(x, (0, 0, 0, sp_pad))
    if not pcp_performance_trace.enabled:
        output = _custom_collective("custom_reduce_scatter", x)
        if output is not None:
            return output
        return tensor_model_parallel_reduce_scatter(x, 0)

    local_payload_bytes = x.numel() * x.element_size()
    with pcp_performance_trace.region(
        "tp.sp_reduce_scatter",
        local_payload_bytes=local_payload_bytes,
        output_payload_bytes=local_payload_bytes // tp_size,
        group_size=tp_size,
    ):
        output = _custom_collective("custom_reduce_scatter", x)
        if output is not None:
            return output
        return tensor_model_parallel_reduce_scatter(x, 0)


def sp_shard(x: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    sp_pad = (-x.shape[0]) % tp_size
    if sp_pad > 0:
        pad = (0, 0) * (x.ndim - 1) + (0, sp_pad)
        x = torch.nn.functional.pad(x, pad)
    chunk = x.shape[0] // tp_size
    return x[tp_rank * chunk : (tp_rank + 1) * chunk]


def sp_padding_mask(
    is_padding: torch.Tensor | None,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    if is_padding is None:
        is_padding = hidden_states.new_zeros(num_tokens, dtype=torch.bool)
    assert is_padding.shape[0] == num_tokens

    tp_size = get_tensor_model_parallel_world_size()
    sp_pad = (-num_tokens) % tp_size
    if sp_pad > 0:
        is_padding = torch.nn.functional.pad(is_padding, (0, sp_pad), value=True)
    chunk = is_padding.shape[0] // tp_size
    tp_rank = get_tensor_model_parallel_rank()
    return is_padding[tp_rank * chunk : (tp_rank + 1) * chunk]
