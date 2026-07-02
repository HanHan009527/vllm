# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, cast

import torch

import vllm.envs as envs
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.common.ops import (
    combine_topk_swa_indices,
    combine_topk_swa_indices_with_positions,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
)
from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
    compute_fp8_einsum_recipe,
    deep_gemm_fp8_o_proj,
)
from vllm.models.deepseek_v4.pcp_metadata import (
    build_pcp_sparse_prefill_rows,
    build_pcp_swa_prefill_segments,
    pcp_swa_torch_sparse_fwd,
)
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4FlashMLAMetadata,
)
from vllm.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
)
from vllm.v1.worker.cp_utils import guard_dsv4_pcp_prefill_runtime_metadata
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

logger = init_logger(__name__)


def _finite_amax_for_diag(tensor: torch.Tensor) -> float:
    finite = torch.isfinite(tensor)
    if not finite.any():
        return float("nan")
    return float(torch.amax(torch.abs(tensor[finite].float())).item())


def _pcp_cache_slot_coverage_diag(
    *,
    k_cache: torch.Tensor,
    write_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    block_size: int,
    chunk_start: int,
    chunk_end: int,
) -> dict[str, int]:
    valid_write_slots = write_slot_mapping[write_slot_mapping >= 0].detach().cpu()
    if valid_write_slots.numel() == 0:
        return {
            "read_slots": 0,
            "write_slots": 0,
            "write_unique": 0,
            "missing": 0,
            "missing_min": -1,
            "missing_max": -1,
            "read_min": -1,
            "read_max": -1,
            "write_min": -1,
            "write_max": -1,
            "scale_min": -1,
            "scale_max": -1,
            "scale_gt200": 0,
            "scale_gt240": 0,
        }

    block_table_cpu = block_table.detach().cpu()
    seq_lens_cpu = seq_lens.detach().cpu()
    gather_lens_cpu = gather_lens.detach().cpu()
    read_slots: list[int] = []
    for req_idx in range(chunk_start, chunk_end):
        seq_len = int(seq_lens_cpu[req_idx].item())
        gather_len = int(gather_lens_cpu[req_idx].item())
        start_pos = seq_len - gather_len
        for pos in range(start_pos, seq_len):
            block_idx = pos // block_size
            pos_in_block = pos % block_size
            physical_block = int(block_table_cpu[req_idx, block_idx].item())
            read_slots.append(physical_block * block_size + pos_in_block)

    if not read_slots:
        read_slots_cpu = torch.empty(0, dtype=torch.long)
    else:
        read_slots_cpu = torch.tensor(read_slots, dtype=torch.long)
    write_unique = torch.unique(valid_write_slots)
    if read_slots_cpu.numel() == 0:
        missing_slots = read_slots_cpu
    else:
        missing_slots = read_slots_cpu[~torch.isin(read_slots_cpu, write_unique)]

    scale_min = -1
    scale_max = -1
    scale_gt200 = 0
    scale_gt240 = 0
    if read_slots_cpu.numel() > 0:
        read_slots_gpu = read_slots_cpu.to(device=k_cache.device, non_blocking=True)
        block_idx = read_slots_gpu // block_size
        pos_in_block = read_slots_gpu % block_size
        cache_2d = k_cache.view(k_cache.shape[0], -1)
        scale_base = block_size * 576 + pos_in_block * 8
        scales = torch.stack(
            [cache_2d[block_idx, scale_base + i] for i in range(7)],
            dim=1,
        )
        scales_cpu = scales.detach().cpu()
        scale_min = int(scales_cpu.min().item())
        scale_max = int(scales_cpu.max().item())
        scale_gt200 = int((scales_cpu > 200).sum().item())
        scale_gt240 = int((scales_cpu > 240).sum().item())

    return {
        "read_slots": int(read_slots_cpu.numel()),
        "write_slots": int(valid_write_slots.numel()),
        "write_unique": int(write_unique.numel()),
        "missing": int(missing_slots.numel()),
        "missing_min": int(missing_slots.min().item())
        if missing_slots.numel()
        else -1,
        "missing_max": int(missing_slots.max().item())
        if missing_slots.numel()
        else -1,
        "read_min": int(read_slots_cpu.min().item()) if read_slots_cpu.numel() else -1,
        "read_max": int(read_slots_cpu.max().item()) if read_slots_cpu.numel() else -1,
        "write_min": int(write_unique.min().item()),
        "write_max": int(write_unique.max().item()),
        "scale_min": scale_min,
        "scale_max": scale_max,
        "scale_gt200": scale_gt200,
        "scale_gt240": scale_gt240,
    }


class DeepseekV4FlashMLAAttention(DeepseekV4Attention):
    """FlashMLA sparse MLA attention layer for DeepSeek V4 (CUDA)."""

    backend_cls = DeepseekV4FlashMLABackend

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._einsum_recipe, self._tma_aligned_scales = compute_fp8_einsum_recipe()

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        force_bf16 = False
        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, dict):
            swa_metadata = cast(
                "DeepseekSparseSWAMetadata | None",
                attn_metadata.get(self.swa_cache_layer.prefix),
            )
            force_bf16 = (
                swa_metadata is not None
                and swa_metadata.pcp_allgather_restore_idx is not None
                and swa_metadata.num_prefill_tokens > 0
            )
        return deep_gemm_fp8_o_proj(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.wo_a,
            self.wo_b,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            o_lora_rank=self.o_lora_rank,
            einsum_recipe=self._einsum_recipe,
            tma_aligned_scales=self._tma_aligned_scales,
            force_bf16=force_bf16,
        )

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        # FP8 decode kernel only supports h_q = 64 or 128.
        if num_heads > 128:
            raise ValueError(
                f"DeepseekV4 FlashMLA does not support {num_heads} heads "
                "(FP8 decode kernel requires h_q in {64, 128})."
            )
        return 64 if num_heads <= 64 else 128

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        # Get SWA and indexer metadata from forward context
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            # Warmup dummy run: no real metadata. Reserve the same bf16
            # gather workspace _forward_prefill would; the dequantize / topk
            # / sparse_fwd kernels are skipped this step.
            swa_only = self.compress_ratio <= 1
            N = (
                0
                if swa_only
                else (self.max_model_len + self.compress_ratio - 1)
                // self.compress_ratio
            )
            M = N + self.window_size + self.max_num_batched_tokens
            current_workspace_manager().get_simultaneous(
                ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            )
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None, attn_metadata.get(self.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        # SWA-only layers (compress_ratio <= 1) don't have their own KV cache
        # allocation, so self.kv_cache may be empty after profiling cleanup.
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        # Split prefill and decode
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                # C4A: local indices differ per layer (filled by Indexer).
                assert self.topk_indices_buffer is not None
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    self.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                # C128A: pre-computed during metadata build.
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens

        # We treat queries in the same seq as different queries
        # and later we only attend by generated indices.
        # q arrives pre-padded to self.padded_heads by the outer wrapper.
        q = q.unsqueeze(1)

        # Prepare SWA cache (num_blocks, swa_block_size, 1, head_bytes)
        # Use unsqueeze to preserve strides (handles padded blocks correctly)
        swa_cache = self.swa_cache_layer.kv_cache.unsqueeze(-2)
        # Reshape KV cache to (num_blocks, block_size, 1, head_bytes)
        if kv_cache is not None:
            kv_cache = kv_cache.unsqueeze(-2)

        # One FlashMLASchedMeta per layer type, shared across all same-type
        # layers within this decode step. The first forward call per type
        # triggers the in-kernel planner (allocating tile_scheduler_metadata
        # and num_splits via PyTorch's graph-aware allocator so CUDA graph
        # capture reuses the same addresses on replay); subsequent same-type
        # layers see have_initialized=True and skip the planner.
        if self.compress_ratio <= 1:
            tile_metadata = swa_metadata.tile_sched_swaonly
        elif self.compress_ratio == 4:
            tile_metadata = swa_metadata.tile_sched_c4a
        elif self.compress_ratio == 128:
            tile_metadata = swa_metadata.tile_sched_c128a
        else:
            raise ValueError(
                f"Unsupported compress_ratio={self.compress_ratio}; "
                "expected 1, 4, or 128."
            )
        assert tile_metadata is not None, (
            "swa_metadata missing tile_sched entry for "
            f"compress_ratio={self.compress_ratio}; "
            "DeepseekSparseSWAMetadataBuilder.build_tile_scheduler did not "
            "allocate one for this layer type."
        )

        out, _ = flash_mla_with_kvcache(
            q=q,
            k_cache=swa_cache,
            block_table=None,
            head_dim_v=512,
            tile_scheduler_metadata=tile_metadata,
            cache_seqlens=None,
            is_fp8_kvcache=True,
            indices=swa_indices,
            topk_length=swa_lens,
            softmax_scale=self.scale,
            attn_sink=self.attn_sink,
            extra_k_cache=kv_cache if not swa_only else None,
            extra_indices_in_kvcache=topk_indices,
            extra_topk_length=topk_lens,
            out=output.unsqueeze(1),
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = attn_metadata is None

        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        is_pcp_prefill = swa_metadata.pcp_allgather_restore_idx is not None
        guard_dsv4_pcp_prefill_runtime_metadata(
            pcp_allgather_restore_idx=swa_metadata.pcp_allgather_restore_idx,
            num_prefill_tokens=num_prefill_tokens,
            runtime_metadata=swa_metadata.pcp_prefill_metadata,
        )

        # Use pre-computed prefill metadata.
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        # Derive prefill-local token offsets from the full query_start_loc_cpu.
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                # C128A: pre-computed during metadata build.
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            top_k = topk_indices.shape[-1]
        else:
            # NOTE(woosuk): topk_indices will not be used for SWA-only layers.
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
        chunk_plan = swa_metadata.get_prefill_chunk_plan(
            compress_ratio=self.compress_ratio,
            prefill_chunk_size=self.PREFILL_CHUNK_SIZE,
        )
        assert chunk_plan, "prefill chunk plan must be non-empty when num_prefills > 0"
        workspace_manager = current_workspace_manager()
        for chunk_start, chunk_end, chunk_N, chunk_M in chunk_plan:
            chunk_size = chunk_end - chunk_start
            kv = workspace_manager.get_simultaneous(
                ((chunk_size, chunk_M, q.shape[-1]), torch.bfloat16),
            )[0]
            if not swa_only:
                # Gather compressed KV
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )

            # Gather SWA KV
            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=chunk_N,
            )

            # Combine the topk indices and SWA indices for gathered KV cache
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            chunk_query_start_loc = query_start_loc[
                num_decodes + chunk_start : num_decodes + chunk_end + 1
            ]
            if swa_metadata.pcp_allgather_restore_idx is not None:
                combined_indices, combined_lens = (
                    combine_topk_swa_indices_with_positions(
                        topk_indices[query_start:query_end],
                        positions[query_start:query_end],
                        chunk_query_start_loc,
                        seq_lens[chunk_start:chunk_end],
                        gather_lens[chunk_start:chunk_end],
                        self.window_size,
                        self.compress_ratio,
                        top_k,
                        chunk_M,
                        chunk_N,
                    )
                )
            else:
                combined_indices, combined_lens = combine_topk_swa_indices(
                    topk_indices[query_start:query_end],
                    chunk_query_start_loc,
                    seq_lens[chunk_start:chunk_end],
                    gather_lens[chunk_start:chunk_end],
                    self.window_size,
                    self.compress_ratio,
                    top_k,
                    chunk_M,
                    chunk_N,
                )
            pcp_sparse_rows = None
            if is_pcp_prefill:
                local_query_start_loc = query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ]
                local_query_start_loc = local_query_start_loc - local_query_start_loc[0]
                pcp_sparse_rows = build_pcp_sparse_prefill_rows(
                    combined_lens=combined_lens,
                    positions=positions[query_start:query_end],
                    local_query_start_loc=local_query_start_loc,
                    seq_lens=seq_lens[chunk_start:chunk_end],
                    gather_lens=gather_lens[chunk_start:chunk_end],
                    chunk_n=chunk_N,
                    chunk_m=chunk_M,
                )
            pcp_diag_layer = self.prefix in (
                "model.layers.0.attn",
                "model.layers.2.attn",
            )
            if envs.VLLM_DSV4_NONFINITE_DIAG and is_pcp_prefill and pcp_diag_layer:
                assert pcp_sparse_rows is not None
                slot_coverage = _pcp_cache_slot_coverage_diag(
                    k_cache=swa_k_cache,
                    write_slot_mapping=swa_metadata.slot_mapping,
                    block_table=swa_block_table,
                    seq_lens=seq_lens,
                    gather_lens=gather_lens,
                    block_size=swa_metadata.block_size,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                )
                valid_offsets = torch.arange(
                    combined_indices.shape[1],
                    device=combined_indices.device,
                    dtype=combined_lens.dtype,
                )
                valid_mask = valid_offsets.unsqueeze(0) < combined_lens.unsqueeze(1)
                valid_indices = combined_indices[valid_mask]
                kv_rows = kv.shape[0] * kv.shape[1]
                invalid_count = (
                    (valid_indices < 0) | (valid_indices >= kv_rows)
                ).sum()
                lens_over_128 = (combined_lens > 128).sum()
                if valid_indices.numel() > 0:
                    indices_min = int(valid_indices.min().item())
                    indices_max = int(valid_indices.max().item())
                else:
                    indices_min = -1
                    indices_max = -1
                logger.error(
                    "DeepSeek V4 PCP sparse prefill diag at %s: "
                    "chunk=(%d,%d) q_tokens=%d positions_min=%d "
                    "positions_max=%d seq_lens=%s gather_lens=%s "
                    "chunk_N=%d chunk_M=%d kv_rows=%d topk=%d "
                    "lens_min=%d lens_max=%d lens_over_128=%d "
                    "indices_min=%d indices_max=%d invalid_indices=%d "
                    "pcp_kernel_rows_min=%d pcp_kernel_rows_max=%d "
                    "pcp_sparse_rows=%d q_finite=%s q_amax=%s "
                    "kv_finite=%s kv_amax=%s "
                    "read_slots=%d write_slots=%d write_unique=%d "
                    "missing_read_slots=%d missing_min=%d missing_max=%d "
                    "read_slot_min=%d read_slot_max=%d "
                    "write_slot_min=%d write_slot_max=%d "
                    "scale_min=%d scale_max=%d scale_gt200=%d "
                    "scale_gt240=%d",
                    self.prefix,
                    chunk_start,
                    chunk_end,
                    query_end - query_start,
                    int(positions[query_start:query_end].min().item()),
                    int(positions[query_start:query_end].max().item()),
                    seq_lens[chunk_start:chunk_end].detach().cpu().tolist(),
                    gather_lens[chunk_start:chunk_end].detach().cpu().tolist(),
                    chunk_N,
                    chunk_M,
                    kv_rows,
                    top_k,
                    int(combined_lens.min().item()),
                    int(combined_lens.max().item()),
                    int(lens_over_128.item()),
                    indices_min,
                    indices_max,
                    int(invalid_count.item()),
                    pcp_sparse_rows.rows_min,
                    pcp_sparse_rows.rows_max,
                    pcp_sparse_rows.sparse_rows,
                    bool(torch.isfinite(q[query_start:query_end]).all().item()),
                    _finite_amax_for_diag(q[query_start:query_end]),
                    bool(torch.isfinite(kv).all().item()),
                    _finite_amax_for_diag(kv),
                    slot_coverage["read_slots"],
                    slot_coverage["write_slots"],
                    slot_coverage["write_unique"],
                    slot_coverage["missing"],
                    slot_coverage["missing_min"],
                    slot_coverage["missing_max"],
                    slot_coverage["read_min"],
                    slot_coverage["read_max"],
                    slot_coverage["write_min"],
                    slot_coverage["write_max"],
                    slot_coverage["scale_min"],
                    slot_coverage["scale_max"],
                    slot_coverage["scale_gt200"],
                    slot_coverage["scale_gt240"],
                )
            if is_pcp_prefill:
                assert pcp_sparse_rows is not None
                if top_k == 0:
                    kv_flat = kv.view(-1, 1, q.shape[-1])
                    segments = build_pcp_swa_prefill_segments(
                        combined_indices=combined_indices,
                        combined_lens=combined_lens,
                        positions=positions[query_start:query_end],
                        local_query_start_loc=local_query_start_loc,
                        seq_lens=seq_lens[chunk_start:chunk_end],
                        gather_lens=gather_lens[chunk_start:chunk_end],
                        chunk_n=chunk_N,
                        chunk_m=chunk_M,
                        window_size=self.window_size,
                    )
                    for segment in segments:
                        seg_output = output[
                            query_start
                            + segment.query_start : query_start
                            + segment.query_end
                        ]
                        if not segment.valid_mask.any():
                            seg_output.zero_()
                            continue

                        seg_kv = kv_flat[segment.kv_start : segment.kv_end]
                        segment_q = q[
                            query_start
                            + segment.query_start : query_start
                            + segment.query_end
                        ]
                        seg_q = segment_q[segment.valid_mask]
                        seg_indices = segment.shifted_indices[segment.valid_mask]
                        seg_lens = segment.topk_lens[segment.valid_mask]
                        seg_out = output.new_empty(
                            (seg_q.shape[0], *output.shape[1:])
                        )
                        seg_out.zero_()
                        pcp_swa_torch_sparse_fwd(
                            q=seg_q,
                            kv=seg_kv,
                            indices=seg_indices,
                            sm_scale=self.scale,
                            attn_sink=self.attn_sink,
                            topk_length=seg_lens,
                            out=seg_out,
                        )
                        seg_out_finite = torch.isfinite(seg_out).all()
                        if (
                            envs.VLLM_DSV4_NONFINITE_DIAG
                            and pcp_diag_layer
                            and (self.prefix == "model.layers.2.attn"
                                 or not bool(seg_out_finite.item()))
                        ):
                            logger.error(
                                "DeepSeek V4 PCP SWA segment diag at %s: "
                                "query=(%d,%d) kv=(%d,%d) sparse_rows=%d "
                                "q_finite=%s kv_finite=%s "
                                "valid_out_finite=%s bad_valid_out=%d "
                                "q_amax=%s kv_amax=%s out_amax=%s "
                                "indices_min=%d indices_max=%d "
                                "lens_min=%d lens_max=%d",
                                self.prefix,
                                segment.query_start,
                                segment.query_end,
                                segment.kv_start,
                                segment.kv_end,
                                segment.sparse_rows,
                                bool(torch.isfinite(seg_q).all().item()),
                                bool(torch.isfinite(seg_kv).all().item()),
                                bool(seg_out_finite.item()),
                                int((~torch.isfinite(seg_out)).sum().item()),
                                _finite_amax_for_diag(seg_q),
                                _finite_amax_for_diag(seg_kv),
                                _finite_amax_for_diag(seg_out),
                                int(seg_indices.min().item()),
                                int(seg_indices.max().item()),
                                int(seg_lens.min().item()),
                                int(seg_lens.max().item()),
                            )
                        seg_output.zero_()
                        seg_output[segment.valid_mask] = seg_out
                    continue
                pcp_q = q.new_zeros((pcp_sparse_rows.sparse_rows, *q.shape[1:]))
                pcp_out = output.new_empty(
                    (pcp_sparse_rows.sparse_rows, *output.shape[1:])
                )
                pcp_out.zero_()
                pcp_indices = combined_indices.new_full(
                    (pcp_sparse_rows.sparse_rows, combined_indices.shape[1]), -1
                )
                pcp_indices[:, 0] = 0
                pcp_lens = combined_lens.new_ones((pcp_sparse_rows.sparse_rows,))

                valid_query_mask = pcp_sparse_rows.valid_query_mask
                if valid_query_mask.any():
                    valid_pcp_kernel_rows = pcp_sparse_rows.q_rows[valid_query_mask]
                    pcp_q.index_copy_(
                        0,
                        valid_pcp_kernel_rows,
                        q[query_start:query_end][valid_query_mask],
                    )
                    pcp_indices.index_copy_(
                        0,
                        valid_pcp_kernel_rows,
                        combined_indices[valid_query_mask],
                    )
                    pcp_lens.index_copy_(
                        0,
                        valid_pcp_kernel_rows,
                        combined_lens[valid_query_mask],
                    )
                flash_mla_sparse_fwd(
                    q=pcp_q,
                    kv=kv.view(-1, 1, q.shape[-1]),
                    indices=pcp_indices.unsqueeze(1),
                    sm_scale=self.scale,
                    attn_sink=self.attn_sink,
                    topk_length=pcp_lens,
                    out=pcp_out,
                )
                chunk_output = output[query_start:query_end]
                chunk_output.copy_(pcp_out.index_select(0, pcp_sparse_rows.q_rows))
                chunk_output[combined_lens == 0] = 0
            else:
                flash_mla_sparse_fwd(
                    q=q[query_start:query_end],
                    kv=kv.view(-1, 1, q.shape[-1]),
                    indices=combined_indices.unsqueeze(1),
                    sm_scale=self.scale,
                    attn_sink=self.attn_sink,
                    topk_length=combined_lens,
                    out=output[query_start:query_end],
                )
