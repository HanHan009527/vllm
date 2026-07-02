# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 prefill PCP metadata helpers."""

from dataclasses import dataclass
from typing import Literal

import torch

from vllm.v1.worker.cp_utils import PCPInterleaveRequestView


@dataclass(frozen=True)
class DeepseekV4PcpSwaSegment:
    query_start: int
    query_end: int
    kv_start: int
    kv_end: int
    sparse_rows: int
    q_rows: torch.Tensor
    shifted_indices: torch.Tensor
    topk_lens: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class DeepseekV4PcpSparseRows:
    q_rows: torch.Tensor
    sparse_rows: int
    valid_query_mask: torch.Tensor
    rows_min: int
    rows_max: int


@dataclass
class DeepseekV4PcpPrefillMetadata:
    cp_size: int
    cp_rank: int
    strategy: Literal["dual_chunk"]
    views: list[PCPInterleaveRequestView]
    local_query_start_loc: torch.Tensor
    local_seq_lens: torch.Tensor
    local_swa_indices: torch.Tensor | None
    local_swa_valid_lens: torch.Tensor | None
    local_c4_indices: torch.Tensor | None
    local_c128_indices: torch.Tensor | None
    global_slot_mapping: torch.Tensor
    compressor_write_locs_global: torch.Tensor | None
    restore_idx: torch.Tensor | None
    debug_global_positions: torch.Tensor | None
    sparse_rows: DeepseekV4PcpSparseRows | None = None
    swa_segments: list[DeepseekV4PcpSwaSegment] | None = None


def pcp_swa_torch_sparse_fwd(
    *,
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    sm_scale: float,
    attn_sink: torch.Tensor | None,
    out: torch.Tensor,
) -> None:
    """Reference sparse SWA attention for PCP prefill segment fallback."""
    out.zero_()
    active_rows = topk_length > 0
    if not active_rows.any():
        return

    q = q[active_rows]
    indices = indices[active_rows]
    topk_length = topk_length[active_rows]

    num_rows, num_heads, _ = q.shape
    max_topk = indices.shape[1]
    valid_offsets = torch.arange(
        max_topk, device=indices.device, dtype=topk_length.dtype
    )
    valid_mask = valid_offsets.unsqueeze(0) < topk_length.unsqueeze(1)
    safe_indices = torch.where(indices >= 0, indices, torch.zeros_like(indices))
    gathered_kv = kv.index_select(0, safe_indices.reshape(-1)).view(
        num_rows, max_topk, kv.shape[1], kv.shape[-1]
    )
    gathered_kv = gathered_kv.squeeze(2).to(torch.float32)
    q_float = q.to(torch.float32)
    scores = torch.einsum("rhd,rkd->rhk", q_float, gathered_kv) * sm_scale
    scores = scores.masked_fill(~valid_mask.unsqueeze(1), -float("inf"))
    if attn_sink is not None:
        sink = attn_sink[:num_heads].to(torch.float32).view(1, num_heads, 1)
        sink = sink.expand(num_rows, -1, -1)
        scores = torch.cat([scores, sink], dim=-1)

    scores = torch.nan_to_num(
        scores,
        nan=torch.finfo(scores.dtype).min,
        posinf=torch.finfo(scores.dtype).max,
        neginf=torch.finfo(scores.dtype).min,
    )
    scores = scores - scores.max(dim=-1, keepdim=True).values
    probs = torch.softmax(scores, dim=-1)
    if attn_sink is not None:
        probs = probs[..., :max_topk]
    out[active_rows].copy_(
        torch.einsum("rhk,rkd->rhd", probs, gathered_kv).to(out.dtype)
    )


def build_pcp_sparse_prefill_rows(
    *,
    combined_lens: torch.Tensor,
    positions: torch.Tensor,
    local_query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    chunk_n: int,
    chunk_m: int,
) -> DeepseekV4PcpSparseRows:
    """Map PCP query rows to compact rows in the gathered KV workspace."""
    q_tokens = int(positions.numel())
    q_rows = torch.empty(q_tokens, dtype=torch.long, device=positions.device)
    chunk_size = int(seq_lens.numel())
    for req_idx in range(chunk_size):
        req_query_start = int(local_query_start_loc[req_idx].item())
        req_query_end = int(local_query_start_loc[req_idx + 1].item())
        if req_query_start == req_query_end:
            continue
        req_seq_len = int(seq_lens[req_idx].item())
        req_gather_len = int(gather_lens[req_idx].item())
        req_gather_start = req_seq_len - req_gather_len
        req_positions = positions[req_query_start:req_query_end].to(torch.long)
        req_valid = (req_positions >= req_gather_start) & (
            req_positions < req_seq_len
        )
        req_rows = req_idx * chunk_m + chunk_n + req_positions - req_gather_start
        req_rows = torch.where(req_valid, req_rows, torch.zeros_like(req_rows))
        q_rows[req_query_start:req_query_end] = req_rows

    valid_query_mask = combined_lens > 0
    rows_min = 0
    rows_max = -1
    if q_tokens > 0 and valid_query_mask.any():
        valid_rows = q_rows[valid_query_mask]
        rows_min = int(valid_rows.min().item())
        rows_max = int(valid_rows.max().item())
        if rows_min < 0 or rows_max >= chunk_size * chunk_m:
            raise ValueError(
                "DeepSeek V4 PCP sparse prefill query rows are outside the "
                "gathered KV workspace: "
                f"min={rows_min}, max={rows_max}, "
                f"workspace_rows={chunk_size * chunk_m}"
            )
        sparse_rows = rows_max + 1
    else:
        sparse_rows = 1
    return DeepseekV4PcpSparseRows(
        q_rows=q_rows,
        sparse_rows=sparse_rows,
        valid_query_mask=valid_query_mask,
        rows_min=rows_min,
        rows_max=rows_max,
    )


def build_pcp_swa_prefill_segments(
    *,
    combined_indices: torch.Tensor,
    combined_lens: torch.Tensor,
    positions: torch.Tensor,
    local_query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    chunk_n: int,
    chunk_m: int,
    window_size: int,
    segment_size: int = 64,
) -> list[DeepseekV4PcpSwaSegment]:
    """Build compact local SWA segments for PCP sparse prefill.

    FlashMLA sparse prefill consumes compact local row coordinates. PCP prefill
    positions are global within the request, so each segment is rebased to the
    smallest local KV window that covers its valid query rows.
    """
    segments: list[DeepseekV4PcpSwaSegment] = []
    chunk_size = int(seq_lens.numel())
    for req_idx in range(chunk_size):
        req_query_start = int(local_query_start_loc[req_idx].item())
        req_query_end = int(local_query_start_loc[req_idx + 1].item())
        req_seq_len = int(seq_lens[req_idx].item())
        req_gather_len = int(gather_lens[req_idx].item())
        req_gather_start = req_seq_len - req_gather_len

        seg_query_start = req_query_start
        while seg_query_start < req_query_end:
            tentative_query_end = min(seg_query_start + segment_size, req_query_end)
            seg_positions_all = positions[
                seg_query_start:tentative_query_end
            ].to(torch.long)
            seg_lens_all = combined_lens[seg_query_start:tentative_query_end]
            seg_valid_all = seg_lens_all > 0
            if seg_positions_all.numel() > 1:
                adjacent_valid = seg_valid_all[:-1] & seg_valid_all[1:]
                position_breaks = adjacent_valid & (
                    seg_positions_all[1:] != seg_positions_all[:-1] + 1
                )
                if position_breaks.any():
                    first_break = int(position_breaks.nonzero()[0].item()) + 1
                    seg_query_end = seg_query_start + first_break
                else:
                    seg_query_end = tentative_query_end
            else:
                seg_query_end = tentative_query_end

            seg_positions = positions[seg_query_start:seg_query_end].to(torch.long)
            seg_lens = combined_lens[seg_query_start:seg_query_end]
            seg_valid = seg_lens > 0
            if not seg_valid.any():
                segments.append(
                    DeepseekV4PcpSwaSegment(
                        query_start=seg_query_start,
                        query_end=seg_query_end,
                        kv_start=0,
                        kv_end=0,
                        sparse_rows=1,
                        q_rows=torch.zeros_like(seg_positions),
                        shifted_indices=combined_indices[
                            seg_query_start:seg_query_end
                        ],
                        topk_lens=seg_lens,
                        valid_mask=seg_valid,
                    )
                )
                seg_query_start = seg_query_end
                continue

            valid_positions = seg_positions[seg_valid]
            seg_base_pos = max(
                req_gather_start,
                int(valid_positions.min().item()) - window_size + 1,
            )
            seg_end_pos = int(valid_positions.max().item()) + 1
            kv_start = req_idx * chunk_m + chunk_n + seg_base_pos - req_gather_start
            kv_end = req_idx * chunk_m + chunk_n + seg_end_pos - req_gather_start
            q_rows = seg_positions - seg_base_pos
            q_rows = torch.where(seg_valid, q_rows, torch.zeros_like(q_rows))
            sparse_rows = int(q_rows[seg_valid].max().item()) + 1

            seg_indices = combined_indices[seg_query_start:seg_query_end]
            shifted_indices = torch.where(
                seg_indices >= 0,
                seg_indices - kv_start,
                seg_indices,
            )
            valid_offsets = torch.arange(
                shifted_indices.shape[1],
                device=shifted_indices.device,
                dtype=seg_lens.dtype,
            )
            valid_index_mask = valid_offsets.unsqueeze(0) < seg_lens.unsqueeze(1)
            valid_shifted_indices = shifted_indices[valid_index_mask]
            if (
                (valid_shifted_indices < 0).any()
                or (valid_shifted_indices >= kv_end - kv_start).any()
            ):
                raise ValueError(
                    "DeepSeek V4 PCP SWA prefill segment indices are outside "
                    "the rebased KV workspace"
                )

            segments.append(
                DeepseekV4PcpSwaSegment(
                    query_start=seg_query_start,
                    query_end=seg_query_end,
                    kv_start=kv_start,
                    kv_end=kv_end,
                    sparse_rows=sparse_rows,
                    q_rows=q_rows,
                    shifted_indices=shifted_indices,
                    topk_lens=seg_lens,
                    valid_mask=seg_valid,
                )
            )
            seg_query_start = seg_query_end
    return segments
