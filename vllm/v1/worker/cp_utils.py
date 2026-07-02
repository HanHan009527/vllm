# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.v1.utils import CpuGpuBuffer

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
else:
    AttentionLayerBase = object


DSV4_PCP_PREFILL_UNSUPPORTED_ERROR = (
    "DeepSeek-V4 prefill PCP requires dsv4 PCP runtime metadata path; "
    "legacy sparse backend remap path is unsupported."
)


@dataclass(frozen=True)
class PCPInterleaveRequestView:
    req_idx: int
    global_seq_len: int
    local_token_count: int
    local_query_start: int
    local_query_end: int
    global_positions: torch.Tensor
    local_positions: torch.Tensor
    restore_idx: torch.Tensor
    global_slot_mapping: torch.Tensor
    local_kv_base: int
    local_kv_len: int


def guard_dsv4_pcp_prefill_runtime_metadata(
    *,
    pcp_allgather_restore_idx: torch.Tensor | None,
    num_prefill_tokens: int,
    runtime_metadata: object | None,
) -> None:
    """Fail closed before DeepSeek V4 sparse backends use legacy PCP remap."""
    if (
        pcp_allgather_restore_idx is not None
        and num_prefill_tokens > 0
        and runtime_metadata is None
    ):
        raise NotImplementedError(DSV4_PCP_PREFILL_UNSUPPORTED_ERROR)


def _cpu_long_tensor(data: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.detach().to(device="cpu", dtype=torch.long)
    return torch.as_tensor(data, dtype=torch.long, device="cpu")


def build_pcp_interleave_request_views(
    *,
    original_token_counts: np.ndarray | torch.Tensor,
    local_token_counts: np.ndarray | torch.Tensor,
    local_positions: np.ndarray | torch.Tensor,
    restore_idx: torch.Tensor,
    pcp_world_size: int,
    global_slot_mapping: torch.Tensor,
    local_valid_mask: np.ndarray | torch.Tensor | None = None,
) -> list[PCPInterleaveRequestView]:
    """Build per-request PCP views from the current local-rank token layout.

    The current V1 PCP manager uses a dual-chunk head/tail layout. The view keeps
    the request-local selected global positions, compact local positions, and the
    per-request restore slice together so model-specific metadata builders do not
    need to rediscover those relationships from raw buffers.
    """
    original_counts = _cpu_long_tensor(original_token_counts)
    local_counts = _cpu_long_tensor(local_token_counts)
    positions = _cpu_long_tensor(local_positions)
    valid_mask = (
        _cpu_long_tensor(local_valid_mask).to(dtype=torch.bool)
        if local_valid_mask is not None
        else None
    )
    slots = global_slot_mapping.detach().to(device="cpu", dtype=torch.long)
    restore = restore_idx.detach().to(device="cpu", dtype=torch.long)

    num_reqs = int(original_counts.numel())
    original_starts = torch.empty(num_reqs, dtype=torch.long)
    local_starts = torch.empty(num_reqs, dtype=torch.long)
    padded_starts = torch.empty(num_reqs, dtype=torch.long)
    if num_reqs == 0:
        return []
    original_starts[0] = 0
    local_starts[0] = 0
    padded_starts[0] = 0
    if num_reqs > 1:
        original_starts[1:] = torch.cumsum(original_counts, dim=0)[:-1]
        local_starts[1:] = torch.cumsum(local_counts, dim=0)[:-1]
        padded_starts[1:] = torch.cumsum(local_counts * pcp_world_size, dim=0)[:-1]

    views: list[PCPInterleaveRequestView] = []
    compact_start = 0
    for req_idx in range(num_reqs):
        global_seq_len = int(original_counts[req_idx].item())
        local_count = int(local_counts[req_idx].item())
        local_start = int(local_starts[req_idx].item())
        local_end = local_start + local_count

        req_positions = positions[local_start:local_end]
        if valid_mask is None:
            req_valid_mask = req_positions < global_seq_len
        else:
            req_valid_mask = valid_mask[local_start:local_end]

        valid_positions = req_positions[req_valid_mask]
        valid_count = int(valid_positions.numel())
        original_start = int(original_starts[req_idx].item())
        slot_indices = original_start + valid_positions
        request_slots = slots[slot_indices] if valid_count > 0 else slots[:0]
        compact_end = compact_start + valid_count

        padded_start = int(padded_starts[req_idx].item())
        padded_end = padded_start + local_count * pcp_world_size
        views.append(
            PCPInterleaveRequestView(
                req_idx=req_idx,
                global_seq_len=global_seq_len,
                local_token_count=valid_count,
                local_query_start=compact_start,
                local_query_end=compact_end,
                global_positions=valid_positions,
                local_positions=torch.arange(valid_count, dtype=torch.long),
                restore_idx=restore[padded_start:padded_end],
                global_slot_mapping=request_slots,
                local_kv_base=compact_start,
                local_kv_len=valid_count,
            )
        )
        compact_start = compact_end
    return views


class PCPManager:
    """Build per-rank token metadata for Prefill Context Parallelism.

    PCP splits long prefill requests across ranks using a head/tail chunk
    assignment. Decode requests are not split; they are replicated on every PCP
    rank so mixed decode+prefill batches keep decode semantics unchanged.

    The manager owns the small CPU/GPU buffers needed by the model runner to:

    * replace scheduled token counts with the local PCP-rank token counts;
    * build the local token positions for this PCP rank;
    * mask padding after PCP all-gather;
    * restore all-gathered hidden/KV tensors back to original request order.
    """

    def __init__(
        self,
        pcp_world_size: int,
        pcp_rank: int,
        max_buffer_num_tokens: int,
        max_num_reqs: int,
        device: torch.device,
        pin_memory: bool = False,
    ) -> None:
        assert pcp_world_size > 1
        assert 0 <= pcp_rank < pcp_world_size
        self.pcp_world_size = pcp_world_size
        self.pcp_rank = pcp_rank

        self.pcp_allgather_restore_idx = CpuGpuBuffer(
            max_buffer_num_tokens,
            dtype=torch.int64,
            device=device,
            pin_memory=pin_memory,
        )
        self.pcp_padded_slot_mapping = torch.empty(
            (max_buffer_num_tokens,),
            dtype=torch.int64,
            device=device,
        )
        self.pcp_padded_positions = torch.empty(
            (max_buffer_num_tokens,),
            dtype=torch.int64,
            device=device,
        )
        self.pcp_padded_query_start_loc = CpuGpuBuffer(
            (max_num_reqs + 1,),
            dtype=torch.int32,
            device=device,
            pin_memory=pin_memory,
        )
        self.num_pcp_pads_cpu_tensor = torch.zeros(
            (max_num_reqs,), device="cpu", dtype=torch.int64
        )
        self.num_pcp_pads_cpu = self.num_pcp_pads_cpu_tensor.numpy()
        self.pcp_unpad_mask = CpuGpuBuffer(
            (max_buffer_num_tokens,),
            dtype=torch.bool,
            device=device,
            pin_memory=pin_memory,
        )
        self.pcp_unpad_mask_cpu_tensor = self.pcp_unpad_mask.cpu
        self.pcp_unpad_mask_gpu_tensor = self.pcp_unpad_mask.gpu
        self.pcp_unpad_mask_cpu = self.pcp_unpad_mask_cpu_tensor.numpy()
        self.pcp_local_unpad_mask = CpuGpuBuffer(
            (max_buffer_num_tokens,),
            dtype=torch.bool,
            device=device,
            pin_memory=pin_memory,
        )
        self.pcp_local_unpad_mask_cpu_tensor = self.pcp_local_unpad_mask.cpu
        self.pcp_local_unpad_mask_gpu_tensor = self.pcp_local_unpad_mask.gpu
        self.pcp_local_unpad_mask_cpu = (
            self.pcp_local_unpad_mask_cpu_tensor.numpy()
        )
        self.pcp_local_token_indices = CpuGpuBuffer(
            max_buffer_num_tokens,
            dtype=torch.int64,
            device=device,
            pin_memory=pin_memory,
        )
        self.pcp_local_token_indices_cpu_tensor = (
            self.pcp_local_token_indices.cpu
        )
        self.pcp_local_token_indices_gpu_tensor = (
            self.pcp_local_token_indices.gpu
        )
        self.pcp_local_token_indices_cpu = (
            self.pcp_local_token_indices_cpu_tensor.numpy()
        )
        self.pcp_request_views: list[PCPInterleaveRequestView] = []

    @staticmethod
    def _get_cumsum_and_arange(
        num_scheduled_tokens: np.ndarray,
        arange_np: np.ndarray,
        cumsum_dtype: np.dtype | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return cumulative token counts and per-request aranges.

        Example: [2, 5, 3] -> ([2, 7, 10],
        [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]).
        """
        cu_num_tokens = np.cumsum(num_scheduled_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        cumsums_offsets = np.repeat(
            cu_num_tokens - num_scheduled_tokens, num_scheduled_tokens
        )
        arange = arange_np[:total_num_tokens] - cumsums_offsets
        return cu_num_tokens, arange

    def update_tokens_for_pcp(
        self,
        tokens: np.ndarray,
        arange_np: np.ndarray,
        num_reqs: int,
        reorder_batch_threshold: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update token counts and positions for this PCP rank.

        Args:
            tokens: Scheduled token counts per request before PCP splitting.
            arange_np: Reusable arange buffer large enough for the padded batch.
            num_reqs: Number of active requests in the prefix of ``tokens``.
            reorder_batch_threshold: Decode/prefill split threshold used by MLA
                metadata builders. Requests with scheduled tokens less than or
                equal to this threshold are treated as decode requests.

        Returns:
            ``(pcp_tokens, pcp_positions)`` for this rank.
        """
        assert reorder_batch_threshold is not None, (
            "PCP depends on reorder batch to split decode and prefill requests."
        )
        tokens = tokens[:num_reqs]
        num_decode_reqs = int(np.sum(tokens <= reorder_batch_threshold))
        num_decode_tokens = int(np.sum(tokens[:num_decode_reqs]))

        num_padded_scheduled_tokens = (
            np.ceil(tokens / (2 * self.pcp_world_size)).astype(np.int32)
            * (2 * self.pcp_world_size)
        )
        num_padded_scheduled_tokens[:num_decode_reqs] = (
            tokens[:num_decode_reqs] * self.pcp_world_size
        )

        self.num_pcp_pads_cpu[:num_reqs] = num_padded_scheduled_tokens - tokens

        cu_padded_tokens, pcp_padded_arange = self._get_cumsum_and_arange(
            num_padded_scheduled_tokens, arange_np
        )
        self.pcp_unpad_mask_cpu[: pcp_padded_arange.shape[0]] = (
            pcp_padded_arange < np.repeat(tokens, num_padded_scheduled_tokens)
        )
        self.pcp_unpad_mask.copy_to_gpu(pcp_padded_arange.shape[0])

        pcp_tokens = num_padded_scheduled_tokens // self.pcp_world_size
        self.pcp_padded_query_start_loc.cpu[0] = 0
        pcp_padded_query_start = np.cumsum(
            pcp_tokens[:num_reqs] * self.pcp_world_size, dtype=np.int32
        )
        self.pcp_padded_query_start_loc.cpu[1 : num_reqs + 1].copy_(
            torch.from_numpy(pcp_padded_query_start)
        )
        self.pcp_padded_query_start_loc.copy_to_gpu(num_reqs + 1)
        pcp_chunk_sizes = (pcp_tokens // 2).clip(min=1)
        pcp_chunk_sizes[:num_decode_reqs] = pcp_tokens[:num_decode_reqs]

        _, pcp_arange = self._get_cumsum_and_arange(pcp_tokens, arange_np)
        _, pcp_chunk_arange = self._get_cumsum_and_arange(
            pcp_chunk_sizes, arange_np
        )
        pcp_head_chunk_mask = pcp_arange < np.repeat(pcp_chunk_sizes, pcp_tokens)

        def get_current_rank_positions(
            positions_start_loc: int | np.ndarray, rank: int
        ) -> np.ndarray:
            positions = np.zeros(len(pcp_head_chunk_mask), dtype=np.int32)
            head_start_loc = positions_start_loc + rank * pcp_chunk_sizes
            tail_start_loc = (
                positions_start_loc
                + (2 * self.pcp_world_size - rank - 1) * pcp_chunk_sizes
            )
            positions[pcp_head_chunk_mask] = pcp_chunk_arange + np.repeat(
                head_start_loc, pcp_chunk_sizes
            )
            positions[~pcp_head_chunk_mask] = (
                pcp_chunk_arange[num_decode_tokens:]
                + np.repeat(tail_start_loc, pcp_chunk_sizes)[num_decode_tokens:]
            )
            return positions

        positions = get_current_rank_positions(0, self.pcp_rank)
        if num_decode_reqs > 0:
            positions[:num_decode_tokens] = self._get_cumsum_and_arange(
                tokens[:num_decode_reqs], arange_np
            )[1]
        original_cu_tokens = np.cumsum(tokens, dtype=np.int64)
        original_start_loc = np.roll(original_cu_tokens, 1)
        original_start_loc[0] = 0
        num_local_tokens = positions.shape[0]
        local_valid_mask = positions < np.repeat(tokens, pcp_tokens)
        self.pcp_local_unpad_mask_cpu[:num_local_tokens] = local_valid_mask
        self.pcp_local_unpad_mask.copy_to_gpu(num_local_tokens)
        self.pcp_local_token_indices_cpu[:num_local_tokens] = (
            positions.astype(np.int64)
            + np.repeat(original_start_loc, pcp_tokens)
        )
        self.pcp_local_token_indices_cpu[:num_local_tokens][
            ~local_valid_mask
        ] = 0
        self.pcp_local_token_indices.copy_to_gpu(num_local_tokens)

        padded_pos_start_loc = np.roll(cu_padded_tokens, 1)
        padded_pos_start_loc[0] = 0
        all_positions = np.concatenate(
            [
                get_current_rank_positions(padded_pos_start_loc, rank)
                for rank in range(self.pcp_world_size)
            ]
        )
        restore_idx = all_positions.argsort()
        self.pcp_allgather_restore_idx.np[: restore_idx.shape[0]] = restore_idx
        self.pcp_allgather_restore_idx.copy_to_gpu(restore_idx.shape[0])
        identity_slot_mapping = torch.arange(
            int(tokens.sum(dtype=np.int64)),
            dtype=torch.long,
            device="cpu",
        )
        self.pcp_request_views = build_pcp_interleave_request_views(
            original_token_counts=tokens,
            local_token_counts=pcp_tokens[:num_reqs],
            local_positions=positions[:num_local_tokens],
            restore_idx=torch.from_numpy(restore_idx),
            pcp_world_size=self.pcp_world_size,
            global_slot_mapping=identity_slot_mapping,
            local_valid_mask=local_valid_mask,
        )

        return pcp_tokens[:num_reqs], positions


def check_attention_cp_compatibility(vllm_config: VllmConfig) -> None:
    pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    interleave_size = vllm_config.parallel_config.cp_kv_cache_interleave_size
    if pcp_size * dcp_size > 1:
        layer_type = cast(type[Any], AttentionLayerBase)
        layers = get_layers_from_vllm_config(vllm_config, layer_type)
        for layer in layers.values():
            layer_impl = getattr(layer, "impl", None)
            if layer_impl is None:
                continue
            if vllm_config.speculative_config is not None and interleave_size > 1:
                assert layer_impl.supports_mtp_with_cp_non_trivial_interleave_size, (
                    "MTP with cp_kv_cache_interleave_size > 1 is not "
                    f"supported in {layer_impl.__class__.__name__}."
                )
            if dcp_size > 1:
                assert layer_impl.need_to_return_lse_for_decode, (
                    "Decode Context Parallelism (DCP) requires attention "
                    "implementations to return the softmax LSE during decode, "
                    f"but {layer_impl.__class__.__name__} does not. "
                    "Try a different backend by setting "
                    "--attention-backend or disable DCP."
                )

            if pcp_size > 1:
                assert layer_impl.supports_pcp, (
                    "PCP requires attention impls' support, "
                    f"but the impl {layer_impl.__class__.__name__} "
                    "does not support PCP."
                )


def get_total_cp_world_size():
    try:
        pcp_world_size = get_pcp_group().world_size
    except AssertionError:
        # PCP might not be initialized in testing
        pcp_world_size = 1
    try:
        dcp_world_size = get_dcp_group().world_size
    except AssertionError:
        # DCP might not be initialized in testing
        dcp_world_size = 1
    return dcp_world_size * pcp_world_size
