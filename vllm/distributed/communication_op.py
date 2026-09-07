# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.distributed

from vllm.pcp_performance import pcp_performance_trace

from .parallel_state import get_tp_group


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    tp_group = get_tp_group()
    if not pcp_performance_trace.enabled:
        return tp_group.all_reduce(input_)
    with pcp_performance_trace.region(
        "tp.all_reduce",
        local_payload_bytes=input_.numel() * input_.element_size(),
        group_size=tp_group.world_size,
    ):
        return tp_group.all_reduce(input_)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    tp_group = get_tp_group()
    if not pcp_performance_trace.enabled:
        return tp_group.all_gather(input_, dim)
    local_payload_bytes = input_.numel() * input_.element_size()
    with pcp_performance_trace.region(
        "tp.all_gather",
        local_payload_bytes=local_payload_bytes,
        gathered_payload_bytes=local_payload_bytes * tp_group.world_size,
        group_size=tp_group.world_size,
        dim=dim,
    ):
        return tp_group.all_gather(input_, dim)


def tensor_model_parallel_reduce_scatter(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""
    tp_group = get_tp_group()
    if not pcp_performance_trace.enabled:
        return tp_group.reduce_scatter(input_, dim)
    local_payload_bytes = input_.numel() * input_.element_size()
    with pcp_performance_trace.region(
        "tp.reduce_scatter",
        local_payload_bytes=local_payload_bytes,
        output_payload_bytes=local_payload_bytes // tp_group.world_size,
        group_size=tp_group.world_size,
        dim=dim,
    ):
        return tp_group.reduce_scatter(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> torch.Tensor | None:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: dict[Any, torch.Tensor | Any] | None = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
